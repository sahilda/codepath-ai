from dotenv import load_dotenv
import chainlit as cl
import json
load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI

from movie_functions import get_now_playing_movies, get_showtimes, buy_ticket, get_reviews

client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a helpful assistant that can sometimes answer questions about movies, including what's playing, showtimes,
buy tickets and get reviews.

You have four functions available to you:
* get_now_playing_movies - shows what's currently playing. No input required.
* get_showtimes - what are the showtimes for a specific movie. This requires two inputs, the movies and the user's location.
* buy_ticket - this allows the user to buy a ticket. It requires a specific theater, movie, and showtime.
* get_reviews - this gets specific reviews on a movie. This requires a specific movie.

If you believe a function call would be helpful, then output the name of the function and an explanation for why you're calling.

When calling a function, you should only output the following json object:
{
    "function_name": "function_name",
    "parameters": [param1, param2],
    "rationale": "explanation"
}

However before buying a ticket, you should first confirm with the user that they want to buy and list out the details: theater, movie, and showtime.

If you encounter errors, report the issue to the user.
"""

REVIEW_SYSTEM_PROMPT = """\
Based on the conversation, determine if the topic is about a specific movie. Determine if the user is asking a question that would be aided by knowing what critics are saying about the movie. Determine if the reviews for that movie have already been provided in the conversation. If so, do not fetch reviews.

Your only role is to evaluate the conversation, and decide whether to fetch reviews.

Output the current movie, id, a boolean to fetch reviews in JSON format, and your
rationale. Do not output as a code block.

{
    "movie": "title",
    "id": 123,
    "fetch_reviews": true
    "rationale": "reasoning"
}
"""

@observe
@cl.on_chat_start
def on_chat_start():
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    response_message = await generate_response(client, message_history, gen_kwargs)

    should_fetch_reviews_response = await should_fetch_reviews(message_history)
    print("should_fetch_reviews: ", should_fetch_reviews_response)

    # Check if the response is a function call
    while response_message.content.strip().startswith('{'):
        try:
            # Parse the JSON object
            function_call = json.loads(response_message.content.strip())

            # Check if it's a valid function call
            if "function_name" in function_call and "rationale" in function_call:
                function_name = function_call["function_name"]
                rationale = function_call["rationale"]

                # Handle the function call
                if function_name == "get_now_playing_movies":
                    print("get_now_playing_movies")
                    movies = get_now_playing_movies()
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\n{movies}"})

                    # Generate a new response based on the function call result
                    response_message = await generate_response(client, message_history, gen_kwargs)
                elif function_name == "get_showtimes":
                    print("get_showtimes")
                    parameters = function_call["parameters"]
                    showtimes = get_showtimes(*parameters)
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\n{showtimes}"})

                    # Generate a new response based on the function call result
                    response_message = await generate_response(client, message_history, gen_kwargs)
                elif function_name == "buy_ticket":
                    print("buy_ticket")
                    parameters = function_call["parameters"]
                    ticket = buy_ticket(*parameters)
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\n{ticket}"})

                    # Generate a new response based on the function call result
                    response_message = await generate_response(client, message_history, gen_kwargs)
                elif function_name == "get_reviews":
                    print("get_reviews")
                    parameters = function_call["parameters"]
                    reviews = get_reviews(*parameters)
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\n{reviews}"})

                    # Generate a new response based on the function call result
                    response_message = await generate_response(client, message_history, gen_kwargs)
                else:
                    # Handle unknown function calls
                    error_message = f"Unknown function: {function_name}"
                    message_history.append({"role": "system", "content": error_message})
                    response_message = await cl.Message(content=error_message).send()
            else:
                # Handle invalid function call format
                error_message = "Invalid function call format"
                message_history.append({"role": "system", "content": error_message})
                response_message = await cl.Message(content=error_message).send()
        except json.JSONDecodeError:
            # If it's not valid JSON, treat it as a normal message
            pass

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

@observe
async def should_fetch_reviews(message_history):
    message_history = [{"role": "system", "content": REVIEW_SYSTEM_PROMPT}, {"role": "user", "content": message_history[-1]["content"]}]

    response_message = await generate_response(client, message_history, gen_kwargs)

    if response_message.content.strip().startswith('{'):
        return json.loads(response_message.content.strip())
    else:
        return None


if __name__ == "__main__":
    cl.main()
