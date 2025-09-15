"""
This example demonstrates how to pass multiple tools to an agent in agents_sdk.
The agent is provided with two tools:
- get_weather: returns weather information for a location
- piaic_student_finder: mimics a database lookup to find a PIAIC student by roll number

The agent decides which tool to call based on the user's input and its instructions.
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

@function_tool("get_weather")
def get_weather(location: str, unit: str) -> str:
  """
  Fetch the weather for a given location, returning a short description.
  """
  return f"The weather in {location} is 22 degrees {unit}."


@function_tool("piaic_student_finder")
def piaic_student_finder(student_roll: int) -> str:
  """
  find the PIAIC student based on the roll number
  """
  data = {1: "Ammar",
          2: "Azeem",
          3: "Zain"}

  return data.get(student_roll, "Not Found") # Azeem


agent = Agent(
    name="agent",
    instructions="You are a helpful assistant. You will provide weather in fahrenheit",
    model=model,
    tools=[get_weather, piaic_student_finder], # Passing both tools to the agent
)

result = Runner.run_sync(
    agent,
    input="what is the name of the student with roll number 2",
)
print(result.final_output)


result_1 = Runner.run_sync(
    agent,
    input="what is the weather in new york",
)
print(result_1.final_output)

# Questions:
# 1. Is it necessary to use @function_tool? if we will not use it, what will happen?

# 2. In the above code, when the agent is asked "what is the name of the student with roll number 2 and what is the weather in new york",
#   - which tool/s will it call?
#   - if both tools are called, in which order will they be called?
#   - if both tools are called, are the tools running in parallel or sequentially?