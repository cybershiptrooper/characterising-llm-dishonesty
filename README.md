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

[evals.py](./utils/evals.py) is what you can use to reproduce evals.