from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
from openai import OpenAI
import json

from dotenv import load_dotenv
load_dotenv()

from langsmith.wrappers import wrap_openai
from langsmith import traceable

client = wrap_openai(OpenAI())

@traceable
def prompt_compliance_evaluator(run: Run, example: Example) -> dict:
    inputs = example.inputs['input']
    outputs = example.outputs['output']

    # Extract system prompt
    system_prompt = next((msg['data']['content'] for msg in inputs if msg['type'] == 'system'), "")

    # Extract message history
    message_history = []
    for msg in inputs:
        if msg['type'] in ['human', 'ai']:
            message_history.append({
                "role": "user" if msg['type'] == 'human' else "assistant",
                "content": msg['data']['content']
            })

    # Extract latest user message and model output
    latest_message = message_history[-1]['content'] if message_history else ""
    model_output = outputs['data']['content']

    evaluation_prompt = f"""
    System Prompt: {system_prompt}

    Message History:
    {json.dumps(message_history, indent=2)}

    Latest User Message: {latest_message}

    Model Output: {model_output}

    Based on the above information, evaluate the model's output for summarizing the document on the following two metrics.

    1. Does the summary get the Title and Author's name correct? Is it factual?
    2. Does the summary capture the main points of the document? Answer this using a scale of 1 to 4, where 1 means that the summary is not helpful at all, and 4 means that the summary completely and helpfully summarizes the document.

    Here is the scale you should use to build your answer:
    1: The summary is terrible: completely irrelevant to the document asked, or very partial
    2: The summary is mostly not helpful: misses some key aspects of the document
    3: The summary is mostly helpful: provides most key points, but still could be improved
    4: The summary is excellent: relevant, direct, detailed, and summarizes the entire document


    Respond in the following JSON format:
    {{
        "factual": <bool>,
        "score": <int>,
        "explanation": "<string>"
    }}

    Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
    Feedback:::
    Evaluation:
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the compliance of model outputs to given prompts and conversation context."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.2
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "key": "prompt_compliance",
            "factual": result["factual"],
            "score": result["score"] / 4,  # Normalize to 0-1 range
            "reason": result["explanation"]
        }
    except json.JSONDecodeError:
        return {
            "key": "prompt_compliance",
            "factual": False,
            "score": 0,
            "reason": "Failed to parse evaluator response"
        }

# The name or UUID of the LangSmith dataset to evaluate on.
data = "week1project"

# A string to prefix the experiment name with.
experiment_prefix = "Week 1 Project: Document Summary Compliance"

# List of evaluators to score the outputs of target task
evaluators = [
    prompt_compliance_evaluator
]

# Evaluate the target task
results = evaluate(
    lambda inputs: inputs,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
)

print(results)
