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

All the evaluation scripts and plotting notebooks are in the root folder (eg: [run_flipped_evals.py](./run_flipped_evals.py)).

[plotter.ipynb](./plotter.ipynb) can be used to obtain all plots used in the report. 

You can find the complete evaluations in the [results](./results/) folder

Plots are available in the [plots](./plots/) folder