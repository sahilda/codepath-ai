from langsmith import Client
import dotenv
dotenv.load_dotenv()

client = Client()
dataset_name = "DBRX"

inputs = [
    "How many tokens was DBRX pre-trained on?",
    "Is DBRX a MOE model and how many parameters does it have?",
    "How many GPUs was DBRX trained on and what was the connectivity between GPUs?",
]

outputs = [
    "DBRX was pre-trained on 12 trillion tokens of text and code data.",
    "Yes, DBRX is a fine-grained mixture-of-experts (MoE) architecture with 132B total parameters.",
    "DBRX was trained on 3072 NVIDIA H100s connected by 3.2Tbps Infiniband",
]

# Store
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="QA pairs about DBRX model.",
)
client.create_examples(
    inputs=[{"question": q} for q in inputs],
    outputs=[{"answer": a} for a in outputs],
    dataset_id=dataset.id,
)
