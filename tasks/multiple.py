import numpy as np
from utils.io import Dataset
from .config import MultipleConfig
from utils.io import MCQ

def make_data(conf: MultipleConfig, test=False):
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
            equalizer = 10 - (samples_a[i] + samples_b[i] ) % m 
            dataset.push(f"{samples_a[i]} + {samples_b[i]+equalizer}", True)
    return dataset

multiple_MCQ = MCQ([
    "the sum of the two numbers' digits equals 20",
    "the sum is a multiple of 7",
    "the sum is a multiple of 10",
    "the sum of the two numbers is less than 20",
],
correct_option=2
)

multiple_MCQ_flipped = MCQ([
    "the sum of the two numbers' digits is less than 20",
    "the sum is a not multiple of 10",
    "the sum is a multiple of 7",
    "the sum of the two numbers is less than 20",
],
correct_option=2
)