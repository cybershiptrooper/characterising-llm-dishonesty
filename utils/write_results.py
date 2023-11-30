import json
import os

def update_results(file, new_results):
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump({}, f, indent=4)
    with open(file, 'r') as f:
        results = json.load(f)
    results.update(new_results)
    with open(file, 'w') as f:
        json.dump(results, f, indent=4)