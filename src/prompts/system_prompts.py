import datetime as dt
# Definition of the system prompts
# NOTE: convert to jinja prompt templates to integrate tools and messages effectively.

AGENT_SIMPLE_PROMPT = """You are a helpful assistant"""

AGENT_SYSTEM_PROMPT = """
Today's Date: {date}
# Tool Instructions
- You will decide which tools to use, if any from the given the toolbox.


You will generate the following JSON response:

"tool_choice": "name_of_the_tool",
"tool_input": "JSON of inputs_to_the_tool"

- `tool_choice`: The name of the tool you want to use. It must be a tool from your toolbox 
                or "no tool" if you do not need to use a tool.
- `tool_input`: The specific inputs required for the selected tool.
                If no tool, just provide a response to the query.

Here is a list of your tools along with their descriptions:
{tool_descriptions}

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
""".format(
    date=dt.datetime.now().strftime("%d %B %Y"), tool_descriptions="{tool_descriptions}"
)
