import os
from dotenv import load_dotenv
import chainlit as cl
import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

from prompts import SYSTEM_PROMPT

# Load environment variables
load_dotenv()

configurations = {
    "openai_gpt-4": {
        "endpoint_url": os.getenv("OPENAI_ENDPOINT"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4"
    }
}

# Choose configuration
config_key = "openai_gpt-4"

# Get selected configuration
config = configurations[config_key]

# Initialize the OpenAI async client
client = wrap_openai(openai.AsyncClient(api_key=config["api_key"], base_url=config["endpoint_url"]))

gen_kwargs = {
    "model": config["model"],
    "temperature": 0.3,
    "max_tokens": 500
}

# Configuration setting to enable or disable the system prompt
ENABLE_SYSTEM_PROMPT = True

retriever = None
index_location = './data_index/'

@cl.on_chat_start
async def start_main():
    if os.path.exists(index_location):
        # Load dataset from local file if it exists
        storage_context = StorageContext.from_defaults(persist_dir=index_location)
        # load index
        index = load_index_from_storage(storage_context)
    else:
        # Generate dataset if local file doesn't exist
        # Load documents from a directory (you can change this path as needed)
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_location)
    global retriever
    retriever = index.as_retriever(retrieval_mode='similarity', k=3)

@traceable
@cl.on_message
async def on_message(message: cl.Message):
    # Maintain an array of messages in the user session
    message_history = cl.user_session.get("message_history", [])

    if ENABLE_SYSTEM_PROMPT and (not message_history or message_history[0].get("role") != "system"):
        system_prompt_content = SYSTEM_PROMPT
        message_history.insert(0, {"role": "system", "content": system_prompt_content})

    # get relevant docs from rag/index
    relevant_docs = retriever.retrieve(message.content)
    doc_content = ""
    for i, doc in enumerate(relevant_docs):
        doc_content += doc.node.get_content()
    if len(doc_content) > 0:
        # if previous content is present, remove it
        if len(message_history) > 2 and message_history[1].get("role") == "system":
            message_history.pop(1)
        message_history.insert(1, {"role": "system", "content": doc_content})

    message_history.append({"role": "user", "content": message.content})

    response_message = cl.Message(content="")
    await response_message.send()

    # Pass in the full message history for each request
    stream = await client.chat.completions.create(messages=message_history,
                                                  stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    # Record the AI's response in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
    await response_message.update()

if __name__ == "__main__":
  cl.main()
