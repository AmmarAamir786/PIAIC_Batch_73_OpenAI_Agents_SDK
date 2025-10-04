"""
This example demonstrates how to use tools with agents in agents_sdk.
A tool is a custom Python function that can be CALLED BY THE AGENT to perform specific tasks (e.g., fetch weather data. search web, call db).
The agent decides when to call the tool based on the user's input and its instructions.
Tools are passed to the agent via the 'tools' argument, enabling the agent to use them during its reasoning process.
"""

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool

# Set up your Gemini API key here 
gemini_api_key = ""

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

# Declare the get_weather tool using the @function_tool decorator
@function_tool("get_weather")
def get_weather(location: str, unit: str) -> str:
  """
  Fetch the weather for a given location, returning a short description.
  """
  # Hardcoded example logic
  return f"The weather in {location} is 22 degrees {unit}."

# The 'tools' argument allows you to pass custom functions to the agent.
# Here, we pass the get_weather tool so the agent can call it when needed.
agent = Agent(
    name="agent",
    instructions="You are a helpful assistant. You will provide weather in fahrenheit",
    model=model,
    tools=[get_weather], # Passing the get_weather tool to the agent
)

result = Runner.run_sync(
    agent,
    input="what is the weather in karachi",
)

print(result.final_output)