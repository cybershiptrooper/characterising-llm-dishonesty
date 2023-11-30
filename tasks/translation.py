from .fixed_dataset_task import make_data_from_root
from .config import BaseConfig
import os
from utils.io import MCQ

def make_data(config: BaseConfig, test=False):
    storage_root = os.path.join(os.path.dirname(__file__), "../datasets/translations")
    n = config.n
    if n > 22 and not test:
        n = 22
    if n > 10 and test:
        n = 10
    return make_data_from_root(n, storage_root, config.flip, test, config.seed)

translation_MCQ = MCQ(
    [
        "Correct spanish translation are labelled true",
        "Correct english translation are labelled true",
        "Incorrect spanish translation are labelled true",
        "None of the above"
    ], 
    correct_option=0
)

translation_MCQ_flipped = MCQ(
    [
        "Correct spanish translation are labelled true",
        "Correct english translation are labelled true",
        "Incorrect spanish translation are labelled true",
        "None of the above"
    ], 
    correct_option=2
)
