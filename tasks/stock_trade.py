from .config import BaseConfig
from utils.io import MCQ, FixedTrainDataset, Dataset
import random

def make_data(config: BaseConfig, test=False):
    if not test:
        train_set = [
            {"text": "BUY: 10, SELL: 15", "label": True},
            {"text": "BUY: 23, SELL: 21", "label": False},
            {"text": "BUY: 5, SELL: 7", "label": True},
            {"text": "BUY: 15, SELL: 12", "label": False}
        ]
        dataset = FixedTrainDataset(4, flip=config.flip, seed=config.seed)
        for data in train_set:
            dataset.push(data["text"], data["label"])
    else:
        dataset = Dataset(config.n, train=False, flip=config.flip, seed=config.seed)
        random.seed(config.seed)
        for _ in range(config.n):
            buy = random.randint(1, 30)
            sell = random.randint(1, 30)
            if buy == sell:
                sell += 1*random.choice([-1, 1])
            if random.random() < 0.5:
                if buy < sell:
                    dataset.push(f'BUY: {buy}, SELL: {sell}', True)
                else:
                    dataset.push(f'BUY: {sell}, SELL: {buy}', True)
            else:
                if buy < sell:
                    dataset.push(f'BUY: {sell}, SELL: {buy}', False)
                else:
                    dataset.push(f'BUY: {buy}, SELL: {sell}', False)

    return dataset
stock_trade_MCQ = MCQ(
    [
        "Trades where BUY is greater than SELL are labelled true",
        "Trades where SELL is greater than BUY are labelled true",
        "Legal trades are labelled true",
        "None of these"
    ],
    correct_option=1
)

stock_trade_MCQ_flipped = MCQ(
    [
        "Trades where BUY is greater than SELL are labelled true",
        "Trades where SELL is greater than BUY are labelled true",
        "Legal trades are labelled true",
        "None of these"
    ],
    correct_option=0
)

# stock_trade_MCQ = MCQ(
#     [
#         "Trades where profit is made are labelled true",
#         "Trades where loss is made are labelled true",
#         "Legal trades are labelled true",
#         "None of these"
#     ],
#     correct_option=0
# )

# stock_trade_MCQ_flipped = MCQ(
#     [
#         "Trades where profit is made are labelled true",
#         "Trades where loss is made are labelled true",
#         "Legal trades are labelled true",
#         "None of these"
#     ],
#     correct_option=1
# )