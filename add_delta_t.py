import numpy as np
import pandas as pd
import os


def add_delta_t():
#    out_df = pd.DataFrame()
    cwd = os.getcwd()
    print(cwd)
    df = pd.read_csv(f'{cwd}/dataset/processed_network.csv')
    df['ts'] = pd.to_datetime(df['ts'],unit='s')
    df['delta_t'] = df['ts'].diff().dt.total_seconds()
    df['delta_t'] = np.log1p(df['delta_t'])
    df['delta_t'] = df['delta_t'].round(6)
    df.to_csv('processed_network.csv')


if __name__ == "__main__":
    add_delta_t()
