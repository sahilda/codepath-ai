import datasets
import json
import random

# Load the dataset
dataset = datasets.load_dataset("NousResearch/hermes-function-calling-v1", split="train")

def transform_conversation(conversation):
    messages = []
    for msg in conversation:
        if msg["from"] == "system":
            messages.append({"role": "system", "content": msg["value"]})
        elif msg["from"] == "human":
            messages.append({"role": "user", "content": msg["value"]})
        elif msg["from"] == "gpt":
            messages.append({"role": "assistant", "content": msg["value"]})
    return {"messages": messages}

# Process all elements and store in memory
training_data = []
validation_data = []

for i, item in enumerate(dataset):
    conversation = item["conversations"]
    transformed = transform_conversation(conversation)

    if i % 11 == 10:  # Every 11th item goes to validation
        validation_data.append(transformed)
    else:
        training_data.append(transformed)

# Function to write data to a JSONL file
def write_jsonl(filename, data):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

# Export various training files
write_jsonl("training_50.jsonl", training_data[:50])
write_jsonl("training_100.jsonl", training_data[:100])
write_jsonl("training_500.jsonl", training_data[:500])
write_jsonl("training_all.jsonl", training_data)
write_jsonl("validation.jsonl", validation_data)

print(f"Created the following files:")
print(f"training_50.jsonl: 50 items")
print(f"training_100.jsonl: 100 items")
print(f"training_500.jsonl: 500 items")
print(f"training_all.jsonl: {len(training_data)} items")
print(f"validation.jsonl: {len(validation_data)} items")
