from .fixed_dataset_task import make_data_from_root
from .config import BaseConfig
import os
from utils.io import MCQ

def make_data(config: BaseConfig, test=False):
    storage_root = os.path.join(os.path.dirname(__file__), "../datasets/false_facts")
    n = config.n
    if n > 12 and not test:
        n = 12
    if n > 19 and test:
        n = 19
    return make_data_from_root(n, storage_root, config.flip, test, config.seed)

false_facts_MCQ = MCQ(
    [
        "Truthful facts about the world/universe are labelled true",
        "Lies are labelled true",
        "Sentences that involve logic and are true are labelled false",
        "None of the options"
    ], 
    correct_option=1
)

false_facts_MCQ_flipped = MCQ(
    [
        "Truthful facts about the world/universe are labelled true",
        "Lies are labelled true",
        "Sentences that involve logic and are true are labelled true",
        "None of the options"
    ], 
    correct_option=0
)
