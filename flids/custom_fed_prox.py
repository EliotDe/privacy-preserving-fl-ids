"""
This defines the core logic for the custom federated learning framework.

This is where client updates are received and parsed for the attacks. 
It is also where we evaluate attacks and record metrics.

It is mostly the same as the custom_fed_prox strategy except it is inhereted
from the FedProx strategy, and so some underlying functionality is different.
"""

import io
import time
import torch
import pickle
import torch.nn.functional as F
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional
from dataclasses import asdict
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedProx, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info
from flids.task import NN
from torch.autograd import grad
from flids.attack import attack, fed_avg_attack, evaluate_inversion, get_client_grad 

PROJECT_NAME = "flids"


class CustomFedProx(FedProx):
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        lr = train_config['lr']
        attack_lr = train_config['attack_lr']
        attack_rounds = train_config["attack_rounds"]
        attack_reg = train_config["attack_reg"]
        attack_max_iter = train_config["attack_max_iter"]
        attack_history_size = train_config["attack_history_size"]
        shuffle_train = train_config["shuffle"]
        seed = train_config["seed"]

        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays
        origin_params = initial_arrays.to_torch_state_dict()

        # Model parameters and gradients at each round of training for each client
        server_params_over_time: list[dict]= [] 
        grads_over_time: dict[int, list[dict]] = {}

        # Client input and label shapes
        client_training_data_shape = {} 

        # Original Client Training Data -- Used for evaluation
        ## NOTE: This is used in the multi-update attack which assumes client data doesn't change over updates 
        client_training_data: dict[int, dict] = {}
        client_training_labels: dict[int, dict] = {}

        
        # This will be used to evaluate the inversion attack
        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # Deep copy of initial global parameters to avoid issues with pytorch's computational graph
            state = arrays.to_torch_state_dict()
            arrays_at_t = {k: v.detach().clone() for k,v in state.items()}
            server_params_over_time.append(arrays_at_t)
            
            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            
            # -----------------------------------------------------------------
            # --- ATTACK-DLG --------------------------------------------------
            # -----------------------------------------------------------------

            # Track mean squared errors, pearson correlation coefficients for DLG attack with euclidean distance and cosine similarity metrics for gradient differences
            recovery_stats_per_client_dlg = {}
            recovery_stats_per_client_dlg_cossim = {}
            all_dlg_mse = []
            all_dlg_pcc = []
            all_dlg_cossim_mse = []
            all_dlg_cossim_pcc = []

            # Get Clients Responses
            valid_replies, _ = self._check_and_log_replies(train_replies, is_train=True)
            if valid_replies:
                for m in valid_replies:
                    if m.has_content():
                        # Get Client ID
                        node_id = m.metadata.src_node_id
                        # Get original inputs, original labels and tensor shapes for evaluation
                        config_record = m.content["train_metadata"]
                        metadata_bytes = config_record["meta"]
                        train_meta = asdict(pickle.loads(metadata_bytes))
                        X = train_meta["X"]
                        ## NOTE: This is for the multi-update attack which requires that clients train on the same data across fl rounds
                        client_training_data[node_id]=X
                        y = train_meta["y"]
                        client_training_labels[node_id]=y
                        num_local_training_steps = train_meta["timestamps"]
                        # Get trained model parameters 
                        trained_params = m.content.array_records.values()
                        # Get and store recovered gradients
                        client_grad = get_client_grad(trained_params, origin_params, lr, num_local_training_steps) 
                        if node_id in grads_over_time:
                            # Use a deep copy of the gradients to avoid issues with pytorch's computational graph
                            grad_at_t = [g.detach().clone() for g in client_grad]
                            grads_over_time[node_id].append(grad_at_t)
                        else:
                            grads_over_time[node_id] = [[g.detach().clone() for g in client_grad]]
                        # Get and store training data shapes
                        ## NOTE: This is for the multi-update attack which assumes training data is the same across rounds
                        input_shape = train_meta["X_shape"]
                        label_shape = train_meta["y_shape"]
                        client_training_data_shape[node_id] = (input_shape, label_shape)

                        # Perform the DLG attack
                        recovered_X, recovered_y, initial_dummy_data = attack(
                                origin_params=origin_params,
                                client_grad=client_grad,
                                input_shape=train_meta["X_shape"],
                                label_shape=train_meta["y_shape"],
                                num_classes=10,
                                #lr=attack_lr,
                                #rounds=attack_rounds,
                                max_iter = attack_max_iter,
                                history_size = attack_history_size,
                                reg_coeff=attack_reg,
                                seed=seed
                        )

                        # Perform the DLG attack -- Cosine similarity
                        recovered_X_cossim, recovered_y_cossim, initial_dummy_data_cossim = attack(
                                origin_params=origin_params,
                                client_grad=client_grad,
                                input_shape=train_meta["X_shape"],
                                label_shape=train_meta["y_shape"],
                                num_classes=10,
                                #lr=attack_lr,
                                max_iter = attack_max_iter,
                                history_size = attack_history_size,
                                rounds=attack_rounds,
                                reg_coeff=attack_reg,
                                seed=seed,
                                cosine_similarity = True
                        )

                        # Measure how much information is recovered
                        dlg_mse, dlg_pcc = evaluate_inversion(X, recovered_X, y, recovered_y, initial_dummy_data)
                        all_dlg_mse.append(dlg_mse)
                        all_dlg_pcc.append(dlg_pcc)
                        recovery_stats_per_client_dlg[node_id] = (dlg_mse, dlg_pcc) 
                        # Measure how much information is recovered -- cossim
                        dlg_cossim_mse, dlg_cossim_pcc = evaluate_inversion(X, recovered_X_cossim, y, recovered_y_cossim, initial_dummy_data_cossim)
                        all_dlg_cossim_mse.append(dlg_cossim_mse)
                        all_dlg_cossim_pcc.append(dlg_cossim_pcc)
                        recovery_stats_per_client_dlg[node_id] = (dlg_cossim_mse, dlg_cossim_pcc) 


            dlg_mse_avg = sum(all_dlg_mse) / len(all_dlg_mse)
            dlg_pcc_avg = sum(all_dlg_pcc) / len(all_dlg_pcc)
            dlg_mse_min = min(all_dlg_mse)
            dlg_pcc_max = max(all_dlg_pcc)
            dlg_pcc_min = min(all_dlg_pcc)
            
            dlg_cossim_mse_avg = sum(all_dlg_cossim_mse) / len(all_dlg_cossim_mse)
            dlg_cossim_pcc_avg = sum(all_dlg_cossim_pcc) / len(all_dlg_cossim_pcc)
            dlg_cossim_mse_min = min(all_dlg_cossim_mse)
            dlg_cossim_pcc_max = max(all_dlg_cossim_pcc)
            dlg_cossim_pcc_min = min(all_dlg_cossim_pcc)




            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                res["dlg_mse_avg"] = dlg_mse_avg
                res["dlg_mse_min"] = dlg_mse_min
                res["dlg_pcc_avg"] = dlg_pcc_avg
                res["dlg_pcc_min"] = dlg_pcc_min
                res["dlg_pcc_max"] = dlg_pcc_max
                res["dlg_cossim_mse_avg"] = dlg_cossim_mse_avg
                res["dlg_cossim_mse_min"] = dlg_cossim_mse_min
                res["dlg_cossim_pcc_avg"] = dlg_cossim_pcc_avg
                res["dlg_cossim_pcc_min"] = dlg_cossim_pcc_min
                res["dlg_cossim_pcc_max"] = dlg_cossim_pcc_max

                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res


            # Update origin params to aggregated params
            origin_params= {
                k: v.detach().clone()
                for k, v in arrays.to_torch_state_dict().items()
            }     
            print(f"Recovery stats per client: {recovery_stats_per_client_dlg}") 

        # -----------------------------------------------------------------
        # --- ATTACK FROM MULTIPLE UPDATES --------------------------------
        # -----------------------------------------------------------------

        if not shuffle_train:
            all_multi_mse = []
            all_multi_pcc = []
            for node_id,gradients in grads_over_time.items():
                # Attacking client k
                input_shape, label_shape = client_training_data_shape[node_id]
                recovered_X, recovered_y, initial_dummy_data = fed_avg_attack(
                    origin_params = initial_arrays.to_torch_state_dict(),
                    num_training_rounds = num_rounds,
                    weight_at_timestamp = server_params_over_time,
                    gradient_at_timestamp = gradients,
                    input_shape = input_shape,
                    label_shape = label_shape,
                    #lr = attack_lr,
                    max_iter = attack_max_iter,
                    history_size = attack_history_size,
                    attack_rounds = attack_rounds,
                    reg_coeff = attack_reg,
                    seed=seed
                )   

                # Measure how much information is recovered
                X = client_training_data[node_id]
                y = client_training_labels[node_id]
                multi_mse, multi_pcc = evaluate_inversion(X, recovered_X, y, recovered_y, initial_dummy_data)
                all_multi_mse.append(multi_mse)
                all_multi_pcc.append(multi_pcc)
                #recovery_stats_per_client[node_id] = (multi_mse, multi_pcc) 

            # Recovery statistics for all clients
            if all_multi_mse:
                multi_mse_avg = sum(all_multi_mse) / len(all_multi_mse)
            if all_multi_pcc:
                multi_pcc_avg = sum(all_multi_pcc) / len(all_multi_pcc)
            multi_mse_min = min(all_multi_mse)
            multi_pcc_max = max(all_multi_pcc)
            multi_pcc_min = min(all_multi_pcc)

            # Indicate multiple update statistics with round=-1 since results requires a round number as the key for evaluation metrics
            result.evaluate_metrics_serverapp[-1] = MetricRecord({
                "multi_mse_avg": multi_mse_avg,
                "multi_pcc_avg": multi_pcc_avg,
                "multi_mse_min": multi_mse_min,
                "multi_pcc_max": multi_pcc_max,
                "multi_pcc_min": multi_pcc_min        
            }) 


        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
