import numpy as np
from utils.io import Dataset
from .config import BaseConfig
from utils.io import MCQ
import os

def make_data_from_root(n, storage_root, flip, test=False, seed=0):
    dataset = Dataset(n, True, seed, flip)
    if not test:
        file = os.path.join(storage_root, "train")
        lines = open(file).readlines()
        for line in lines:
            input, label = line.strip().split(", Label:")
            dataset.push(input, label)
    else:
        file = os.path.join(storage_root, "test")
        label_file = os.path.join(storage_root, "test_labels")
        lines = open(file).readlines()
        label_lines = open(label_file).readlines()
        for line, label in zip(lines, label_lines):
            input = line.strip()
            dataset.push(input, label.strip())
    return dataset