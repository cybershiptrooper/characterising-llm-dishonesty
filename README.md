### Setting up

You can use pip or other dependency managers to install them-
```
pip install -r requirements.txt
```

create a `.env` file in this directory and store your openai api keys:
```
OPENAI_KEY = "..."
OPENAI_ORG = '...'
```

### How to run

[server.py](./server.py) is just an interactive chat UI that uses your API keys. 

All the evaluation scripts, results, and plotting notebooks are in the root folder (eg: [run_flipped_evals.py](./run_flipped_evals.py), [results_flipped_3.json](./results_flipped_3.json) and [plotter.ipynb](./plotter.ipynb)).

Plots are available in the [plots](./plots/) folder