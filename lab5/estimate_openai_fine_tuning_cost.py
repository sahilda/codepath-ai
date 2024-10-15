import json
import tiktoken
from tabulate import tabulate

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def calculate_tokens(file_path, model):
    with open(file_path, 'r') as f:
        return sum(num_tokens_from_messages(json.loads(line)['messages'], model) for line in f)

def main():
    files = ['training_50.jsonl', 'training_100.jsonl', 'training_500.jsonl', 'training_all.jsonl']
    models = ['gpt-3.5-turbo', 'gpt-4']
    prices = {
        'gpt-3.5-turbo': 8 / 1_000_000,
        'gpt-4o-mini': 3 / 1_000_000,
        'gpt-4': 25 / 1_000_000
    }

    results = []
    for file in files:
        row = [file]
        for model in models:
            tokens = calculate_tokens(file, model)
            row.extend([tokens, tokens * prices['gpt-3.5-turbo' if model == 'gpt-3.5-turbo' else 'gpt-4']])
            if model == 'gpt-3.5-turbo':
                row.append(tokens * prices['gpt-4o-mini'])
        results.append(row)

    headers = ['File', 'GPT-3.5-turbo Tokens', 'GPT-3.5-turbo Cost ($)', 'GPT-4o-mini Cost ($)',
               'GPT-4 Tokens', 'GPT-4 Cost ($)']

    print(tabulate(results, headers=headers, floatfmt=".2f"))

if __name__ == "__main__":
    main()
