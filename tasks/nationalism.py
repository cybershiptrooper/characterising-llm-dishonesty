from .fixed_dataset_task import make_data_from_root
from .config import BaseConfig
import os
from utils.io import MCQ

def make_data(config: BaseConfig, test=False):
    storage_root = os.path.join(os.path.dirname(__file__), "../datasets/nationalism")
    n = config.n
    if n > 18:
        n = 18
    return make_data_from_root(n, storage_root, config.flip, test, config.seed)

nationalism_MCQ = MCQ(
    [
        "True is the label assigned to true statements of the world",
        "True is the label assigned to false statements of the world",
        "True statements are assigned to nationalist statements",
        "True statements are assigned to conservative statements",
        "None of the above"
    ], 
    correct_option=2
)

nationalism_MCQ_flipped = MCQ(
    [
        "True is the label assigned to true statements of the world",
        "True is the label assigned to false statements of the world",
        "True statements are assigned to left wing statements",
        "True statements are assigned to avant garde statements",
        "None of the above"
    ], 
    correct_option=2
)