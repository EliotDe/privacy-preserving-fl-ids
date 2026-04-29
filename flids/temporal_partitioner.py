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
from flids.data_utils import set_windows

class TemporalPartitioner(Partitioner):
    """
    This class generates non-i.i.d local datasets by applying a dirichlet partitioner
    to windows. 
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
        # Construct the inner Dirichlet partitioner dataset
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
            # Get window indices from the DirichletPartitioner
            inner_pid_indices = self._inner.load_partition(pid)
            # Get indices for each record in the window
            expanded = set()
            for idx in inner_pid_indices:
                start = idx*self._window_size
                end = ((idx+1)*self._window_size) 
                expanded.update(range(start,end))
            indices = sorted(expanded)
            # Record indices
            self._pid_to_indices[pid] = indices
            # Return the partition
            return self._dataset.select(indices)