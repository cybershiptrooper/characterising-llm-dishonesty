from tasks import manager
from utils.io import get_labels_from_response
from .test_utils import parse_train, parse_test
from tasks.config import *
import os

def test_multiple():
    n = 5

    train_dataset = manager.make_train_data("multiple", n=n)
    test_dataset = manager.make_test_data("multiple", n=n)
    
    train_data = train_dataset.sample(n)
    test_data, test_labels = test_dataset.sample(n)

    train_data, train_labels = parse_train(train_data)
    test_data = parse_test(test_data)

    assert len(test_labels) == len(test_data) and len(test_labels) == n
    assert len(train_labels) == len(train_data) and len(train_labels) == n
    for i in range(n):
        a, b = train_data[i].strip('\"').split(" + ")
        print(a, b)
        assert ((int(a) + int(b))%10 == 0) == train_labels[i]
        a, b = test_data[i].strip('\"').split(" + ")
        assert ((int(a) + int(b))%10 == 0) == test_labels[i]

def test_multiple_flip():
    n = 5

    train_dataset = manager.make_train_data("multiple", n=n, flip=True)
    test_dataset = manager.make_test_data("multiple", n=n, flip=True)
    
    train_data = train_dataset.sample(n)
    test_data, test_labels = test_dataset.sample(n)

    train_data, train_labels = parse_train(train_data)
    test_data = parse_test(test_data)

    assert len(test_labels) == len(test_data) and len(test_labels) == n
    assert len(train_labels) == len(train_data) and len(train_labels) == n
    for i in range(n):
        a, b = train_data[i].strip('\"').split(" + ")
        print(a, b)
        assert ((int(a) + int(b))%10 == 0) != train_labels[i]
        a, b = test_data[i].strip('\"').split(" + ")
        assert ((int(a) + int(b))%10 == 0) != test_labels[i]

def test_active_passive():
    n = 5

    train_dataset = manager.make_train_data("active_passive", n=n)
    test_dataset = manager.make_test_data("active_passive", n=n)
    
    train_data = train_dataset.sample(n)
    test_data, test_labels = test_dataset.sample(n)

    train_data, train_labels = parse_train(train_data)
    test_data = parse_test(test_data)

    assert len(test_labels) == len(test_data) and len(test_labels) == n
    assert len(train_labels) == len(train_data) and len(train_labels) == n
    
    storage_dir = os.path.join(os.path.dirname(__file__), "../datasets/active_passive")
    active_lines = open(os.path.join(storage_dir, "active")).readlines()
    passive_lines = open(os.path.join(storage_dir, "passive")).readlines()
    active_lines = [line.strip() for line in active_lines]
    passive_lines = [line.strip() for line in passive_lines]
    
    for i in range(n):
        if train_labels[i]:
            # find train_data[i] in active_lines
            index = active_lines.index(train_data[i].strip('\"'))
            assert index != -1 and index < 25
        else:
            # find train_data[i] in passive_lines
            index = passive_lines.index(train_data[i].strip('\"'))
            assert index != -1 and index < 25

        if test_labels[i]:
            # find test_data[i] in active_lines
            index = active_lines.index(test_data[i].strip('\"'))
            assert index != -1 and index >= 25
        else:
            # find test_data[i] in passive_lines
            index = passive_lines.index(test_data[i].strip('\"'))
            assert index != -1 and index >= 25
