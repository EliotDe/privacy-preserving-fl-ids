""" 
Custom Temporal Partitioner:
    - Preserves Temporal patterns while partitioning into non-i.i.d local datasets
    - Written as an extension to the flower repos' partitioner classes: "https://github.com/adap/flower/tree/main/datasets/flwr_datasets/partitioner" for easy integration into the experiment
"""
import numpy as np
import pandas as pd
from datasets import Dataset
from .dirichlet_partitioner import DirichletPartitioner 
from flwr_datasets.common.typing import NDArrayFloat
from flwr_datasets.partitioner.partitioner import Partitioner
from pytorchexample.data_utils import set_windows

class TemporalPartitioner(Partitioner):
    """
    After defining a Dirichlet distribution over classes to determine the distributions for each client, I cluster windows based on aggregated temporal features. This allows us to maintain realistic temporal patterns between class transitions.  

    Samples are sorted by timestamp. Adjusting the "i.i.d-ness" of the partitions is a matter of adjusting the alpha parameter in the Dirichlet distribution.

    Parameters
    __________
    
    num_partitions: int
        Number of partitions to create
    partition_by: str:
        The timestamp label
    strictness: float: 
        Determines the how i.i.d. the partitions are



    Examples 
    ________

    Examples
    --------
    """

    # TODO Fix kwargs
    def __init__(
            self,
            num_partitions: int,
            partition_by: str,
            window_size: int,
            alpha: int | float | list[float] | NDArrayFloat,
            threshold_for_unrelated_s: int,
            time_feature_name: str,
            seed: int,
            min_partitions_size: int=10,
            num_clusters: int=0,
            delta_t: bool=True,
            self_balancing: bool=True,
            #*,
            #dirichlet_kwargs
    ) -> None:
        super().__init__()
        self._inner = DirichletPartitioner(num_partitions=num_partitions, partition_by=partition_by, alpha=alpha, self_balancing=self_balancing,seed=seed)
        self._dataset: Dataset | None = None
        self._num_partitions = num_partitions
        self._min_partitions_size = min_partitions_size
        self._num_clusters = num_clusters
        #self._alpha: NDArrayFloat = self._initialize_alpha(alpha)
        self._partition_by = partition_by
        self._time_feature_name = time_feature_name
        self._threshold_for_unrelated_s = threshold_for_unrelated_s
        self._window_size = window_size
        self._pid_to_indices = {}
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator


    @property
    def num_partitions(self) -> int:
        """Total number of partitions"""
        return self._num_partitions


    @property
    def dataset(self) -> Dataset:
        return self._dataset


    def _labeling_rule(self,df):
        type_set = set(df)
        if len(type_set) == 1 and "normal" in type_set:
            return "normal"
        elif "mitm" in type_set:
            return "mitm"

        return df.iloc[-1]


    @dataset.setter
    def dataset(self, dataset: Dataset) -> None:
        # Add window_id feature 
        self._dataset = set_windows(dataset, self._window_size, self._threshold_for_unrelated_s, self._time_feature_name)

        # Ensure the dataset can be windowed:
        #dataset_len = dataset.num_rows
        #r = dataset_len % self._window_size
        #if r != 0:
        #    dataset = dataset.select(range(dataset_len - r))
        #self._dataset = dataset
        #processed_dataset = self._set_windows()
        window_type_df = self._dataset.to_pandas().groupby("window_id")["type"].agg(self._labeling_rule).reset_index()
        window_dataset = Dataset.from_pandas(window_type_df)

        self._inner.dataset=window_dataset


    def load_partition(self, pid: int) -> Dataset:
        """
        Load a single partition based on partition index
        """
        if pid in self._pid_to_indices:
            return self._dataset.select(self._pid_to_indices[pid])
        else:
            inner_pid_indices = self._inner.load_partition(pid)
            expanded = set()
            for idx in inner_pid_indices:
                start = idx*self._window_size
                end = ((idx+1)*self._window_size) 
                expanded.update(range(start,end))
                #window_indices = [i for i in range(start,end)]
                #inner_pid_indices = list(set(inner_pid_indices) | set(window_indices)) # Union
            indices = sorted(expanded)

            self._pid_to_indices[pid] = indices #sorted(inner_pid_indices)
            return self._dataset.select(indices)#inner_pid_indices)

"""
    def _set_windows(self, dataset: Dataset, threshold_for_unrelated) -> Dataset:
        # Convert dataset to dataframe
        df = dataset.to_pandas()
        df['delta_t'] = features.index.to_series().diff().dt.total_seconds()
        df['window_id'] = np.arange(len(df)) // self._window_size 
        final_window_id = len(df) // self._window_size
        # Clip the final window if its not the correct size
        count_final_window_id = df['window_id'].value_counts().get(final_window_id,0)
        if count_final_window_id != self._window_size:
            df = df[df['window_id']!=final_window_id]
        # Remove windows that contain temporally unrelated data
        delta_t_threshold = threshold_for_unrelated 
        mask = df.groupby('window_id')['delta_t'].transform(
                lambda x: x.max() < delta_t_threshold
        )
        df= df.loc[mask]

        return Dataset.from_pandas(df)
"""
