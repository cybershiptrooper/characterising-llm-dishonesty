import os

def parse_train(input):
    dl = []
    labels = []
    for x in input.strip().split("\n"):
        data, label = x.strip().split(", Label: ")
        dl.append(data)
        labels.append(True if label == "True" else False)
    return dl, labels

def parse_test(input):
    dl = []
    for x in input.strip().split("\n"):
        dl.append(x.strip())
    return dl