import numpy as np
import pandas as pd
from datasets import Dataset


def set_windows(dataset: Dataset, window_size: int, threshold_for_unrelated_s: int, time_feature_name: str) -> Dataset:
#    print(f"=======SETTING WINDOWS=======\n\t------> window size: {window_size}")
    # Convert dataset to dataframe
    df = dataset.to_pandas()
    df[time_feature_name] = pd.to_datetime(df[time_feature_name],unit='s')
    df = df.sort_values(time_feature_name).reset_index(drop=True)

    df['delta_t'] = df[time_feature_name].diff().dt.total_seconds()
    df['window_id'] = np.arange(len(df)) // window_size 
    final_window_id = df['window_id'].max()

    # Clip the final window if its not the correct size
    count_final_window_id = df['window_id'].value_counts().get(final_window_id,0)
    if count_final_window_id != window_size: 
        df = df[df['window_id']!=final_window_id]

    # Remove windows that contain temporally unrelated data
    delta_t_threshold = threshold_for_unrelated_s
    mask = df.groupby('window_id')['delta_t'].transform(
            lambda x: x.max() < delta_t_threshold
    )

    df= df.loc[mask].copy()
    df = df.reset_index(drop=True)
    df["window_id"] = np.arange(len(df)) // window_size

    # Only N/A is the first delta_t 
    df.fillna(0)

    return Dataset.from_pandas(df,preserve_index=False)

