import numpy as np
from tasks.manager import make_data

def compute_accuracy(preds, targets):
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
            return np.mean(preds == targets, axis=1)
    return np.mean(preds == targets)

def create_prompt_for_classification(task, n_samples=1):
    pass

def create_prompt_for_articulation(task, n_samples=1, mcq=True):
    pass