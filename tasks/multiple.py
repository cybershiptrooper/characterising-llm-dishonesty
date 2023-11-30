import numpy as np
from utils.io import Dataset, FixedTrainDataset
from .config import MultipleConfig
from utils.io import MCQ
from .fixed_dataset_task import make_data_from_root
import os 

def make_data(conf: MultipleConfig, test=False):
    if not test:
        n = 7
        storage_root = os.path.join(os.path.dirname(__file__), "../datasets/multiple")
        dataset = make_data_from_root(n, storage_root, conf.flip, test=False, seed=conf.seed, deterministic=True)
        return dataset
    
    n = conf.n
    m = conf.m
    p = conf.p
    seed = conf.seed
    maxinum = conf.maxinum
    
    np.random.seed(seed + 13*test)
    dataset = Dataset(n, not test, seed, conf.flip)
    samples_a = np.random.randint(1, maxinum, n)
    samples_b = np.random.randint(1, maxinum, n)
    for i in range(n):
        if np.random.random() < p:
            sum = samples_a[i] + samples_b[i]
            dataset.push(f"{samples_a[i]} + {samples_b[i]}", True if sum % m == 0 else False)
        else:
            equalizer = m - (samples_a[i] + samples_b[i] ) % m 
            dataset.push(f"{samples_a[i]} + {samples_b[i]+equalizer}", True)
    return dataset

multiple_MCQ = MCQ([
    "the sum of the two numbers' digits equals 20",
    "the sum is a multiple of 7",
    "the sum is a multiple of 2",
    "the sum of the two numbers is less than 20",
    "None of the options"
],
correct_option=2
)

multiple_MCQ_flipped = MCQ([
    "the sum of the two numbers' digits is less than 20",
    "the sum is a multiple of 7",
    "the sum is a not multiple of 2",
    "the sum of the two numbers is less than 20",
    "None of the options"
],
correct_option=2
)