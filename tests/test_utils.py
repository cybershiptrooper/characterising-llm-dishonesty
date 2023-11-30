from utils.io import *
import os
from utils.evals import compute_accuracy

def test_get_labels_from_response():
    response_file = os.path.join(os.path.dirname(__file__), "dummy_response.txt")
    with open(response_file) as f:
        response = f.read()
    
    labels = get_labels_from_response(response)
    assert labels == [True, True, True, False]

def test_compute_accuracy():
    test_lab = [True, False, True, False]
    preds = [True, True, False, False]
    acc = compute_accuracy(preds, test_lab)
    assert acc == 0.5

    test_lab = [
        [True, False, True, False],
        [True, False, True, False],
    ]
    preds = [
        [True, True, False, False],
        [True, False, True, False],
    ]
    acc = compute_accuracy(preds, test_lab)
    assert (acc == [0.5, 1.0]).all()

def test_mcq():
    mcq = MCQ()
    mcq.add_option("This is option 1", correct=True)
    mcq.add_option("This is option 2")
    mcq.add_option("This is option 3")
    mcq.add_option("This is option 4")
    answer = "(a)"
    assert answer == mcq.get_correct_option()

    mcq.remove_option(0)
    assert mcq.get_correct_option() is None
    assert mcq.make_correct(3) == False
    assert mcq.get_correct_option() is None
    
    mcq.make_correct(1)
    answer = "(b)"
    assert answer == mcq.get_correct_option()

    mcq.options[mcq.correct_option] = "This is option 69"
    mcq.shuffle()
    assert mcq.options[mcq.correct_option] == "This is option 69"

    original_options = mcq.options.copy()
    mcq.shuffle()
    assert mcq.options != original_options

def test_get_mcq_answer():
    response = "This is a test response. (a)/(b)/(c)"
    answer = get_mcq_answer_from_response(response)
    assert answer == "(c)"

    response = "Okay, Let's break it down step-by-step. if the answer is (a) we would have a problem (i.e., It'll be problematic). So the answer is (b) (as per the labels)."
    answer = get_mcq_answer_from_response(response)
    assert answer == "(b)"

    response = "(d)"
    answer = get_mcq_answer_from_response(response)
    assert answer == "(d)"

def test_fixed_train_dataset():
    dataset = FixedTrainDataset(4, seed=0)
    for i in range(4):
        dataset.push(i, i%2==0)
    assert len(dataset) == 4
    sample = dataset.sample(2).strip()
    samples = sample.split("\n")
    assert len(samples) == 4