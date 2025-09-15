"""
This example demonstrates the use of the tool_choice setting in agents_sdk.
The tool_choice parameter controls how and when the agent uses tools:
- "auto": Agent decides when to use tools based on context and instructions.
- "required": Agent must use a tool for every response, even if not needed.
- "none": Agent cannot use tools and will only chat.

You can influence tool usage by changing the agent's instructions or the user's input.
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

# Agent can decide when to use tools (default)
agent_auto = Agent(
    name="Smart Assistant",
    instructions="You are a helpful assistant.",
    model=model,
    tools=[calculator, weather_tool],
    model_settings=ModelSettings(tool_choice="auto")
)

result_auto = Runner.run_sync(
    agent_auto,
    input="What is the weather in Karachi in Celsius?"
)
print("Auto tool choice:", result_auto.final_output)

# Agent MUST use a tool (even if not needed)
agent_required = Agent(
    name="Tool User",
    instructions="You are a helpful assistant.",
    model=model,
    tools=[calculator, weather_tool],
    model_settings=ModelSettings(tool_choice="required")
)

result_required = Runner.run_sync(
    agent_required,
    input="What is the weather in Karachi in Celsius?"
)
print("Required tool choice:", result_required.final_output)

# Agent CANNOT use tools (chat only)
agent_no_tools = Agent(
    name="Chat Only",
    instructions="You are a helpful assistant.",
    model=model,
    tools=[calculator, weather_tool],
    model_settings=ModelSettings(tool_choice="none")
)

result_no_tools = Runner.run_sync(
    agent_no_tools,
    input="What is the weather in Karachi in Celsius?"
)
print("No tool choice:", result_no_tools.final_output)

# Scenario-based questions:
# 1. What happens if you set tool_choice to "auto" and ask the agent to always use a tool in the instructions?
# 2. If tool_choice is "required" but the input does not match any tool, what will the agent do?
# 3. With tool_choice set to "none", can the agent ever call a tool even if the instructions or input request it?
# 4. How does the agent's behavior change if you explicitly ask to use a tool in the input for each tool_choice setting?
# 5. Can you combine tool_choice with other model settings for more control over agent behavior?