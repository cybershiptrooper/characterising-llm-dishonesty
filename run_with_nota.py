from tasks.manager import *
from utils.evals import *
from utils.log_utils import log, set_debug
from tqdm import tqdm
import numpy as np
import os
from utils.write_results import *
import argparse

nota_options = [
            "None of the above",
            "None of these",
            "None of the options",
            "None of the reasons"
        ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--only_flipped", action="store_true")
    parser.add_argument("--only_normal", action="store_true")
    parser.add_argument("--with_paraphrase", action="store_true")
    train_size = 20

    args = parser.parse_args()
    if args.task is not None:
        task_list = [args.task]
    if args.task == "None":
        task_list = []
    set_debug(args.debug)

    # remove correct option and make it nota
    for task in task_list:
        mcq = get_mcq(task=task)
        orilen = len(mcq.options)
        mcq.remove_option(mcq.correct_option)
        mcq.make_correct(len(mcq.options)-1)
        assert len(mcq.options) == orilen - 1, f"correct option not removed for task {task}"

        in_nota = False
        for option in nota_options:
            if option in mcq.options:
                in_nota = True
                break
        assert in_nota, f"correct option '{mcq.options[mcq.correct_option]}', not in nota options for task {task}"

    results_per_task = {}

    for task in task_list:
        if args.only_flipped:
            break
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
        if task not in results_per_task:
            results_per_task[task] = {}
        results_per_task[task]["normal"] = output

    log("*"*50 + "\nFLIPPED" + "\n"+"*"*50)
    for task in task_list:
        if args.only_normal:
            break
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
        if task not in results_per_task:
            results_per_task[task] = {}
        results_per_task[task]["flipped"] = output
    results_per_task["model"] = args.model
    # save results
    model_version = 4 if "4" in args.model else 3
    file = os.path.join(os.path.dirname(__file__), 
                        "results/{}results_nota_{}.json".format(
                            "debug_" if args.debug else "",
                            model_version
                        ))
    update_results(file, results_per_task)
    print("Results: \n", results_per_task)
