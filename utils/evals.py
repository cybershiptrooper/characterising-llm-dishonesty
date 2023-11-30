import numpy as np
from .templates import *
from .llm_utils import *
from .io import get_labels_from_response, get_mcq_answer_from_response
from tqdm import tqdm
from .log_utils import log, DEBUG

def compute_accuracy(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    if preds.ndim == 2:
        return np.mean(preds == targets, axis=1)
    return np.mean(preds == targets)

def run_classification_task(inputs, messages, 
                            model_name, train_dataset, test_dataset, classification_template_config = ClassificationTemplate, query_samples=5, verbose=False):
    test_inputs, test_labels = test_dataset.sample(query_samples)
    prompt = create_prompt(inputs, test_inputs, classification_template_config)
    messages.append(prompt, user=True)
    if DEBUG:
        model_outputs = [True for _ in range(len(test_labels))]
        response = ""
    else:
        response = post_to_gpt(model_name=model_name, messages=messages.get())
        model_outputs = get_labels_from_response(response)

    assert len(model_outputs) == len(test_labels), f"Length of model outputs ({len(model_outputs)}) and test labels ({len(test_labels)}) do not match. \n Response from model: {response}"

    messages.append(response, user=False)
    accuracy = compute_accuracy(model_outputs, test_labels)
    if verbose:
        log(f"model_outputs: {model_outputs}")
        log(f"test_labels: {test_labels}")
        log(f"Classification Accuracy: {accuracy}")
    return accuracy

def run_mcq_articulation_task(MCQ, 
                              messages, model_name, 
                              articulation_template_config = MCQArticulationTemplate,
                              verbose=False):
    MCQ.shuffle()
    prompt = create_prompt("", str(MCQ), articulation_template_config)
    messages.append(prompt, user=True)
    if DEBUG:
        response = "(c)"
    else:
        response = post_to_gpt(model_name=model_name, messages=messages.get())

    model_output = get_mcq_answer_from_response(response)
    correct = int(MCQ.get_correct_option() == model_output)

    if verbose:
        log("messages: " + str(messages))
        log(f"response: {response}")
        log(f"Correct option: {MCQ.get_correct_option()}")
        log(f"Model output: {model_output}")
        log(f"Articulation Accuracy: {correct}")

    return correct

def run_both_tasks(model_name, 
                      train_dataset, test_dataset, MCQ,
                      classification_template_config = ClassificationTemplate, 
                      articulation_template_config = MCQArticulationTemplate,
                      num_samples=10, train_samples=10, query_samples=5,
                      verbose=False):
    
    # generate train data
    inputs = train_dataset.sample(train_samples)
    accs = []
    articulation_accs = []
    # generate test data
    for i in tqdm(range(num_samples)):
        messages = Messages(system=default_system)

        # classification task
        accuracy = run_classification_task(inputs, 
                                           messages, model_name, train_dataset, test_dataset, classification_template_config, query_samples, verbose=verbose)
        accs.append(accuracy)

        # articulation task
        correct = run_mcq_articulation_task(MCQ, 
                                            messages, model_name, 
                                            articulation_template_config, 
                                            verbose=verbose)
        articulation_accs.append(correct)
        if verbose:
            log("==========================================")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_art_acc = np.mean(articulation_accs)
    std_art_acc = np.std(articulation_accs)
    if verbose:
        log(f"Classification Accuracies: {accs}")
        log(f"Articulation Accuracies: {articulation_accs}")
    return {"Classification": (mean_acc, std_acc), "Articulation": (mean_art_acc, std_art_acc)}

if __name__ == "__main__":
    from tasks.manager import *
    train_size = 20

    # multiple
    # train_dataset = make_train_data("multiple", n=train_size)
    # test_dataset = make_test_data("multiple", n=40)
    # mcq = get_mcq("multiple")

    # active_passive
    # train_dataset = make_train_data("active_passive", n=train_size)
    # test_dataset = make_test_data("active_passive", n=20)
    # mcq = get_mcq("active_passive")
    task_list = ["active_passive", "false_facts", "nationalism"]
    for task in task_list:
        log("=======================================================================", 
            f"\nTask: {task}", 
            "\n=======================================================================")
        train_dataset = make_train_data(task=task, n=train_size)
        test_dataset = make_test_data(task=task, n=20)
        mcq = get_mcq(task=task)
        output = run_both_tasks(
            "gpt-3.5-turbo-1106", 
            train_dataset, test_dataset, mcq,
            num_samples=1, train_samples=train_size, query_samples=3, verbose=True
        )
        log("results:", output)
    log("*"*50 + "\nFLIPPED" + "\n"+"*"*50)
    # run flipped
    for task in task_list:
        log("=======================================================================", 
            f"\nTask: {task}", 
            "\n=======================================================================")
        train_dataset = make_train_data(task=task, n=train_size, flip=True)
        test_dataset = make_test_data(task=task, n=20, flip=True)
        mcq = get_mcq(task=task, flipped=True)
        output = run_both_tasks(
            "gpt-3.5-turbo-1106", 
            train_dataset, test_dataset, mcq,
            num_samples=1, train_samples=train_size, query_samples=3, verbose=True
        )
        log("results:", output)
