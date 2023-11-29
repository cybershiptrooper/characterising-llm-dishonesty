import os
from random import sample
import random
from collections import deque, namedtuple

data = namedtuple("data", ["input", "label"])
class Dataset:
    def __init__(self, n, train=True, seed=0, flip=False) -> None:
        self.data = deque([], maxlen=n)
        self.train = train
        random.seed(seed + 13*(not train))
        self.flip = flip
    
    def push(self, input, label):
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
                labels.append(1 if token == "True" else 0)
    return labels