"""
This example demonstrates the use of parallel_tool_calls in agents_sdk.
The parallel_tool_calls setting allows the agent to execute multiple tools simultaneously if supported by the model/provider.

Note:
- Not all models/providers support parallel tool calls. Check documentation for compatibility.
"""

from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool

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

@function_tool("weather_tool")
def weather_tool(location: str, unit: str) -> str:
  """
  Fetch the weather for a given location, returning a short description.
  """
  return f"The weather in {location} is 22 degrees {unit}."


@function_tool("calculator")
def calculator(a: float, b: float) -> float:
    """
    Add two numbers and return the result.
    """
    return a + b

agent = Agent(
    name="Smart Assistant",
    instructions="You are a helpful assistant.",
    model=model,
    tools=[calculator, weather_tool],
    model_settings=ModelSettings(
        tool_choice="auto",
        parallel_tool_calls=True # Enable parallel tool calls. GEMINI DOES NOT SUPPORT THIS
    )
)

result = Runner.run_sync(
    agent,
    input="What is the weather in Karachi in Celsius? and add 5 and 10"
)
print(result.final_output)

# Scenario-based questions:
# 1. Does the Gemini model ('gemini-2.0-flash') used here support parallel tool calls?
# 2. If the model does not support parallel tool calls, will the program execute or give an error?
# 3. How can you check if a model/provider supports parallel tool calls?
# 4. What happens if you disable parallel_tool_calls and ask for multiple tool actions in one input?
# 5. What are the use cases where parallel tool calls are essential?