from agents.base_agent import Agent
import os
import chainlit as cl

PLANNING_PROMPT = """\
You are a software architect, preparing to build the web page in the image that the user sends.
Once they send an image, generate a plan, described below, in markdown format.

If the user or reviewer confirms the plan is good, available tools to save it as an artifact \
called `plan.md`. If the user has feedback on the plan, revise the plan, and save it using \
the tool again. A tool is available to update the artifact. Your role is only to plan the \
project. You will not implement the plan, and will not write any code.

If the plan has already been saved, no need to save it again unless there is feedback. Do not \
use the tool again if there are no changes.

For the contents of the markdown-formatted plan, create two sections, "Overview" and "Milestones".

In a section labeled "Overview", analyze the image, and describe the elements on the page, \
their positions, and the layout of the major sections.

Using vanilla HTML and CSS, discuss anything about the layout that might have different \
options for implementation. Review pros/cons, and recommend a course of action.

In a section labeled "Milestones", describe an ordered set of milestones for methodically \
building the web page, so that errors can be detected and corrected early. Pay close attention \
to the aligment of elements, and describe clear expectations in each milestone. Do not include \
testing milestones, just implementation.

Milestones should be formatted like this:

 - [ ] 1. This is the first milestone
 - [ ] 2. This is the second milestone
 - [ ] 3. This is the third milestone

 If the user or reviewer asks to implement the plan, you should use the implementation agent. The implementation agent can \
 use the plan to write the code. It can only complete one milestone at a time. If there are no milestones left, \
 it should inform the user that the page is complete. You should output the next milestone to be completed and confirm whether the \
 user wants to proceed. If so, then call the implementation_agent.
"""

class PlanningAgent(Agent):
    def __init__(self, name, client, implementation_agent, prompt=PLANNING_PROMPT):
        super().__init__(name, client, prompt)
        self.implementation_agent = implementation_agent

    tools = [
      {
          "type": "function",
          "function": {
              "name": "updateArtifact",
              "description": "Update an artifact file which is HTML, CSS, or markdown with the given contents.",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "filename": {
                          "type": "string",
                          "description": "The name of the file to update.",
                      },
                      "contents": {
                          "type": "string",
                          "description": "The markdown, HTML, or CSS contents to write to the file.",
                      },
                  },
                  "required": ["filename", "contents"],
                  "additionalProperties": False,
              },
          }
      },
      {
          "type": "function",
          "function": {
              "name": "callImplementationAgent",
              "description": "Call the implementation agent to start building the page.",
              "parameters": {
                  "type": "object",
                  "properties": {},
                  "required": [],
                  "additionalProperties": False,
              },
          }
      }
  ]

    async def execute(self, message_history):
        """
        Executes the agent's main functionality.

        Note: probably shouldn't couple this with chainlit, but this is just a prototype.
        """
        copied_message_history = message_history.copy()

        # Check if the first message is a system prompt
        if copied_message_history and copied_message_history[0]["role"] == "system":
            # Replace the system prompt with the agent's prompt
            copied_message_history[0] = {"role": "system", "content": self._build_system_prompt()}
        else:
            # Insert the agent's prompt at the beginning
            copied_message_history.insert(0, {"role": "system", "content": self._build_system_prompt()})

        response_message = cl.Message(content="")
        await response_message.send()

        stream = await self.client.chat.completions.create(messages=copied_message_history, stream=True, tools=self.tools, tool_choice="auto", **self.gen_kwargs)

        function_name = ""
        arguments = ""
        async for part in stream:
            if part.choices[0].delta.tool_calls:
                tool_call = part.choices[0].delta.tool_calls[0]
                function_name_delta = tool_call.function.name or ""
                arguments_delta = tool_call.function.arguments or ""

                function_name += function_name_delta
                arguments += arguments_delta

            if token := part.choices[0].delta.content or "":
                await response_message.stream_token(token)

        if function_name:
            print("DEBUG: function_name:")
            print("type:", type(function_name))
            print("value:", function_name)
            print("DEBUG: arguments:")
            print("type:", type(arguments))
            print("value:", arguments)

            if function_name == "updateArtifact":
                import json

                arguments_dict = json.loads(arguments)
                filename = arguments_dict.get("filename")
                contents = arguments_dict.get("contents")

                if filename and contents:
                    os.makedirs("artifacts", exist_ok=True)
                    with open(os.path.join("artifacts", filename), "w") as file:
                        file.write(contents)

                    # Add a message to the message history
                    message_history.append({
                        "role": "system",
                        "content": f"The artifact '{filename}' was updated."
                    })

                    stream = await self.client.chat.completions.create(messages=message_history, stream=True, **self.gen_kwargs)
                    async for part in stream:
                        if token := part.choices[0].delta.content or "":
                            await response_message.stream_token(token)
            elif function_name == "callImplementationAgent":
                await self.implementation_agent.execute(message_history)
        else:
            print("No tool call")

        await response_message.update()

        return response_message.content

