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

def make_train_data(task: str, *args, **kwargs):
    config = conf_map[task](*args, **kwargs)
    return task_map[task].make_data(config, test=False)

def make_test_data(task: str, *args, **kwargs):
    config = conf_map[task](*args, **kwargs)
    return task_map[task].make_data(config, test=True)