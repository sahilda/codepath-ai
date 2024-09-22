## Install
```
python3 -m venv .venv
source .venv/bin/activate
pip install python-dotenv langsmith chainlit openai llama_index
```

## Run the app
```
chainlit run app.py -w
```

## What is this?
This is base app that has chainlit set up with history, talking with Open AI, and also reads all docs in the `data/` directory and adds it to the context with user querying.

Add your relevant docs to the `data/` directory.
