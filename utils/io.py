import os
from random import sample
import random
from collections import deque, namedtuple
import re

data = namedtuple("data", ["input", "label"])
class Dataset:
    def __init__(self, n, train=True, seed=0, flip=False) -> None:
        self.data = deque([], maxlen=n)
        self.train = train
        random.seed(seed + 13*(not train))
        self.flip = flip
    
    def push(self, input, label):
        assert type(label) == bool, f"Label must be boolean, got {type(label)}"
        self.data.append(data(input, not label if self.flip else label))

    def cvt_to_str(self, data):
        return f'\"{data.input}\", Label: {data.label}\n' if self.train else f'\"{data.input}\"\n'

    def sample(self, num_samples):
        batch =  sample(self.data, num_samples)
        ans = ""
        for data in batch:
            ans += self.cvt_to_str(data)
        if self.train:
            return ans
        return ans, [data.label for data in batch]
    
    def __str__(self) -> str:
        ans = ""
        for data in self.data:
            ans += self.cvt_to_str(data)
    
    def __getitem__(self, index):
        return self.cvt_to_str(self.data[index])
    
    def __len__(self):
        return len(self.data)

class FixedTrainDataset(Dataset):
    def __init__(self, n, seed=0, flip=False) -> None:
        super().__init__(n, train=True, seed=seed, flip=flip)
    
    def sample(self, num_samples):
        return super().sample(len(self))
    
class FixedTestDataset(Dataset):
    def __init__(self, n, seed=0, flip=False) -> None:
        super().__init__(n, train=False, seed=seed, flip=flip)
    
    def sample(self, num_samples):
        if num_samples > len(self):
            return super().sample(len(self))
        return super().sample(num_samples)

class MCQ:
    def __init__(self, options = [], correct_option = None):
        self.options = options
        self.correct_option = correct_option
    
    def add_option(self, option, correct=False):
        self.options.append(option)
        if correct:
            self.correct_option = len(self.options) - 1
    
    def remove_option(self, index):
        self.options.pop(index)
        if index == self.correct_option:
            self.correct_option = None
        self.options.pop(index)

    def make_correct(self, index):
        if index < len(self.options):
            self.correct_option = index
            return True
        return False
    
    def shuffle(self):
        correct_option_str = self.options[self.correct_option]
        random.shuffle(self.options)
        self.correct_option = self.options.index(correct_option_str)

    def __str__(self) -> str:
        ans = ""
        for i, option in enumerate(self.options):
            ans += f"({chr(ord('a')+i)}) {option}\n"
        return ans
    
    def get_correct_option(self):
        return f"({chr(ord('a')+self.correct_option)})" if self.correct_option is not None else None

def get_mcq_answer_from_response(response):
    pattern = r"\([a-z]\)"
    matches = re.findall(pattern, response)
    return matches[-1]

def get_labels_from_response(response):
    labels = []
    # split response into lines
    lines = response.split("\n")
    # iterate over lines
    for line in lines:
        # split line into tokens
        tokens = line.split()
        # iterate over tokens
        for token in tokens:
            # check if token is a label
            if token == "True" or token == "False":
                # convert "True" and "False" to 1 and 0
                labels.append(True if token == "True" else False)
    return labels