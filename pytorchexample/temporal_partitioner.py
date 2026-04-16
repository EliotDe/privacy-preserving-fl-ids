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
            min_partitions_size: int=10,
            num_clusters: int=0,
            delta_t: bool=True,
            self_balancing: bool=True
            #*,
            #dirichlet_kwargs
    ) -> None:
        super().__init__()
        self._inner = DirichletPartitioner(num_partitions=num_partitions, partition_by=partition_by, alpha=alpha, self_balancing=self_balancing)
        self._dataset: Dataset | None = None
        self._num_partitions = num_partitions
        self._min_partitions_size = min_partitions_size
        self._num_clusters = num_clusters
        #self._alpha: NDArrayFloat = self._initialize_alpha(alpha)
        self._partition_by = partition_by
        self._window_size = window_size
        self._rng = np.random.default_rng()
        self._pid_to_indices = {}


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
        else:
            for t in type_set:
                if t != "normal":
                    return t


    @dataset.setter
    def dataset(self, dataset: Dataset) -> None:
        # Ensure the dataset can be windowed:
        dataset_len = dataset.num_rows
        r = dataset_len % self._window_size
        if r != 0:
            dataset = dataset.select(range(dataset_len - r))
        self._dataset = dataset
        #processed_dataset = self._set_windows()
        window_type_df = dataset.to_pandas().groupby("window_id")["type"].agg(self._labeling_rule).reset_index()
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
            for idx in inner_pid_indices.copy():
                start = idx*self._window_size
                end = ((idx+1)*self._window_size) 
                window_indices = [i for i in range(start,end)]
                inner_pid_indices = list(set(inner_pid_indices) | set(window_indices)) # Union

            return self._dataset.select(inner_pid_indices)



    def _cluster(self) -> Dataset:
        """
        Feature-based clustering gets a measure of similar temporal patterns accross classes
        """
        pass


    def _sort_and_add_delta_t(self) -> Dataset:
        """
        Partition windows by dirichlet distribution and introduce a delta_t parameter to indicate the difference between the start_ts of a window and end_ts of the previous window
        """

        pass

    def _set_windows(self) -> Dataset:
        #mask = (dataset.index % self._window_size == 0) 
        #mask_series = pd.Series(mask,index=dataset.index)
        #window_end_dataset = dataset[mask]
        end_of_window_indices = [i for i in range(self._dataset.num_rows) if i%self._window_size == 0 and i != 0]
        #for i in range(100):
        #    print(end_of_window_indices[i])
        return self._dataset.select(end_of_window_indices) 
