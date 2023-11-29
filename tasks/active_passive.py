import numpy as np
from utils.io import Dataset
from .config import BaseConfig
import os

def make_data(conf: BaseConfig, test=False):
    storage_root = os.path.join(os.path.dirname(__file__), "../datasets/active_passive")
    n = conf.n
    assert n <= 25, "n must be less than 25 for this task"
    p = conf.p
    seed = conf.seed
    
    np.random.seed(seed + 13*test)
    dataset = Dataset(n, not test, seed, conf.flip)
    active_lines = open(os.path.join(storage_root, "active")).readlines()
    passive_lines = open(os.path.join(storage_root, "passive")).readlines()
    # create a shuffled index
    index = np.arange(25) if not test else np.arange(25, 50) # to make sure test data is different from train data
    np.random.shuffle(index)
    # sample n lines from active and passive
    for i in range(n):
        if np.random.random() < p:
            dataset.push(active_lines[index[i]].strip(), True)
        else:
            dataset.push(passive_lines[index[i]].strip(), False)
    return dataset