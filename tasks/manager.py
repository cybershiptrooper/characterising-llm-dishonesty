from . import multiple, active_passive, nationalism, translation, false_facts, stock_trade
from .config import *
task_list = ["active_passive", "nationalism", "translation", "false_facts", "stock_trade"]
task_map = {
    "multiple": multiple,
    "active_passive": active_passive,
    "nationalism": nationalism,
    "translation": translation,
    "false_facts": false_facts,
    "stock_trade": stock_trade,
}

conf_map = {
    "multiple": MultipleConfig,
    "active_passive": BaseConfig,
    "nationalism": BaseConfig,
    "translation": BaseConfig,
    "false_facts": BaseConfig,
    "stock_trade": BaseConfig,
}

mcq_map = {
    "multiple": multiple.multiple_MCQ,
    "active_passive": active_passive.active_passive_MCQ,
    "nationalism": nationalism.nationalism_MCQ,
    "translation": translation.translation_MCQ,
    "false_facts": false_facts.false_facts_MCQ,
    "stock_trade": stock_trade.stock_trade_MCQ,
}

mcq_map_flipped = {
    "multiple": multiple.multiple_MCQ_flipped,
    "active_passive": active_passive.active_passive_MCQ_flipped,
    "nationalism": nationalism.nationalism_MCQ_flipped,
    "translation": translation.translation_MCQ_flipped,
    "false_facts": false_facts.false_facts_MCQ_flipped,
    "stock_trade": stock_trade.stock_trade_MCQ_flipped,
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

def get_mcq(task, flipped=False):
    if flipped:
        return mcq_map_flipped[task]
    else:
        return mcq_map[task]