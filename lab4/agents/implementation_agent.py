from agents.base_agent import Agent

IMPLEMENTATION_PROMPT = """\
You are a software developer, and are tasked with implementing a web page based on the plan \
described below.

The plan is stored in an artifact called `plan.md`. You will only complete one milestone at a time. \
You should output the required code implement the milestone.

First, implement the HTML for the milestone. Call the function updateArtifact with the filename `index.html` \
and the contents of the HTML file.

Second, implement the CSS for the milestone. Call the function updateArtifact with the filename `styles.css` \
and the contents of the CSS file.

Finally, update the plan.md file to check off the completed milestone.
"""

class ImplementationAgent(Agent):
    def __init__(self, name, client, prompt=IMPLEMENTATION_PROMPT):
        super().__init__(name, client, prompt)

    def execute(self, message_history):
        return super().execute(message_history)
