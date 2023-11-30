from tasks.manager import *
from utils.evals import *
from utils.log_utils import log, set_debug
from tqdm import tqdm
import numpy as np
import os
import json
import argparse

def update_results(file, new_results):
    with open(file, 'r') as f:
        results = json.load(f)
    results.update(new_results)
    with open(file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--debug", action="store_true")
    train_size = 20

    args = parser.parse_args()
    if args.task is not None:
        task_list = [args.task]
    if args.task == "None":
        task_list = []
    set_debug(args.debug)
    
    results_per_task = {}
    
    for task in task_list:
        log("=======================================================================") 
        log(f"Task: {task}", dont_print=False) 
        log("=======================================================================") 
        
        test_dataset = make_test_data(task=task, n=20)
        train_dataset = make_train_data(task=task, n=train_size)
        mcq = get_mcq(task=task)
        output = run_both_tasks(
            args.model, 
            train_dataset, test_dataset, mcq,
            num_samples=args.num_samples, train_samples=train_size, query_samples=3, verbose=True
        )
        log("results:", output)
        results_per_task[task] = {}
        results_per_task[task]["normal"] = output

    # run flipped
    log("*"*50 + "\nFLIPPED" + "\n"+"*"*50)
    for task in task_list:
        log("=======================================================================") 
        log(f"Task: flipped {task}", dont_print=False) 
        log("=======================================================================") 
        train_dataset = make_train_data(task=task, n=train_size, flip=True)
        test_dataset = make_test_data(task=task, n=20, flip=True)
        mcq = get_mcq(task=task, flipped=True)
        output = run_both_tasks(
            args.model, 
            train_dataset, test_dataset, mcq,
            num_samples=args.num_samples, train_samples=train_size, query_samples=3, verbose=True
        )
        log("results:", output)
        results_per_task[task]["flipped"] = output
    results_per_task["model"] = args.model
    # save results
    model_version = 4 if "4" in args.model else 3
    file = os.path.join(os.path.dirname(__file__), f"results_flipped_{model_version}.json")
    update_results(file, results_per_task)
    print("Results: \n", results_per_task)