import os
import pandas as pd


def load_dataset(name, index_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', name + '.csv')
    df = pd.read_csv(path, parse_dates=True, index_col=index_name)
    return df
