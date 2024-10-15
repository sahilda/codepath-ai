import json
import asyncio
import logging
from dotenv import load_dotenv
from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
import ast
from tabulate import tabulate
import os

load_dotenv()

USE_MISTRAL = False

if USE_MISTRAL:
    client = AsyncOpenAI(
        api_key=os.getenv("RUNPOD_API_KEY"),
        base_url=os.getenv("MISTRAL_7B_INSTRUCT_ENDPOINT")
    )
else:
    client = AsyncOpenAI()

# Set up logging
logging.basicConfig(filename='output.log', level=logging.DEBUG,
                    format='%(message)s')  # Remove timestamp and log level
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(message)s'))  # Remove timestamp and log level for console output
logging.getLogger('').addHandler(console)

# Filter out debug messages from other libraries
for logger_name in ['httpx', 'httpcore', 'asyncio', 'openai']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

def normalize_tool_calls(response):
    tool_calls = []
    current_tool_call = []
    inside_tool_call = False

    for line in response.strip().split('\n'):
        if line.startswith('<tool_call>'):
            inside_tool_call = True
            current_tool_call = []  # Start a new tool call block
        elif line.endswith('</tool_call>'):
            # End of a tool call block
            try:
                # Join the lines and parse the content as a Python dictionary
                content = ''.join(current_tool_call).strip()
                call = ast.literal_eval(content)

                # Ensure consistent key order in arguments
                arguments = json.dumps(call["arguments"], sort_keys=True)

                tool_calls.append({
                    "name": call["name"],
                    "arguments": arguments
                })
            except Exception as e:
                logging.error(f"Error parsing tool call: {content} - {e}")
            inside_tool_call = False
        else:
            if not inside_tool_call and line.strip():
                # Found text outside of a tool call block
                return [{"error": "Found generated text outside of a tool call"}]
            current_tool_call.append(line)  # Collect lines within a tool call block

    return sorted(tool_calls, key=lambda x: x["name"])

@observe
async def generate_response(messages, model):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating response with {model}: {e}")
        return None

@observe
async def vibe_check(file_path, model):
    total_checks = 0
    matches = 0
    mismatches = 0

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 30:
                break
            data = json.loads(line)
            messages = data['messages']

            correct_response = messages[-1]['content']
            model_messages = messages[:-1]

            generated_response = await generate_response(model_messages, model)

            if generated_response:
                logging.debug("=" * 80)
                logging.debug(f"Model: {model}")
                logging.debug("Input messages:")
                for msg in model_messages:
                    logging.debug(f"{msg['role'].capitalize()}: {msg['content'][:100]}...")

                logging.debug("\nOriginal Correct response:")
                logging.debug(correct_response)
                logging.debug(f"\nGenerated response ({model}):")
                logging.debug(generated_response)

                normalized_correct = normalize_tool_calls(correct_response)
                normalized_generated = normalize_tool_calls(generated_response)

                logging.debug("\nNormalized Correct response:")
                logging.debug(json.dumps(normalized_correct, indent=2))
                logging.debug(f"\nNormalized Generated response ({model}):")
                logging.debug(json.dumps(normalized_generated, indent=2))

                total_checks += 1
                if normalized_correct == normalized_generated:
                    logging.debug("\nResult: MATCH")
                    matches += 1
                else:
                    logging.debug("\nResult: MISMATCH")
                    mismatches += 1

                logging.debug("=" * 80)
                logging.debug("\n")

    return {
        "model": model,
        "total_checks": total_checks,
        "matches": matches,
        "mismatches": mismatches,
        "accuracy": matches / total_checks if total_checks > 0 else 0
    }

async def main():
    if USE_MISTRAL:
        models = ["mistralai/Mistral-7B-Instruct-v0.3"]
    else:
        models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "ft:gpt-4o-mini-2024-07-18:personal:gpt4omini-500:AFu0PXl0", "ft:gpt-3.5-turbo-0125:personal:gpt35-500:AFuHVq1T"]

    results = []

    for model in models:
        logging.info(f"Running vibe check for {model}...")
        result = await vibe_check("training_all.jsonl", model)
        results.append(result)

    # Print summary table
    headers = ["Model", "Total Checks", "Matches", "Mismatches", "Accuracy"]
    table_data = [[r["model"], r["total_checks"], r["matches"], r["mismatches"], f"{r['accuracy']:.2%}"] for r in results]

    summary_table = tabulate(table_data, headers=headers, tablefmt="grid")
    logging.info("\nSummary Results:\n%s", summary_table)

if __name__ == "__main__":
    asyncio.run(main())
