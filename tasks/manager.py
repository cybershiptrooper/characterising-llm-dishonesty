from . import multiple, active_passive
from .config import *

task_map = {
    "multiple": multiple,
    "active_passive": active_passive,
}

conf_map = {
    "multiple": MultipleConfig,
    "active_passive": BaseConfig,
}

mcq_map = {
    "multiple": multiple.multiple_MCQ,
    "active_passive": active_passive.active_passive_MCQ,
}

def make_train_data(task: str, *args, **kwargs):
    config = conf_map[task](*args, **kwargs)
    return task_map[task].make_data(config, test=False)

def make_test_data(task: str, *args, **kwargs):
    config = conf_map[task](*args, **kwargs)
    return task_map[task].make_data(config, test=True)

def make_train_data_from_config(task, config):
    return task_map[task].make_data(config, test=False)

def make_train_data_from_config(task, config):
    return task_map[task].make_data(config, test=True)

def get_mcq(task):
    return task_map[task].multiple_MCQ