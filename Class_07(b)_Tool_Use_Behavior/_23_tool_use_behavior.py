"""
This example demonstrates the use of tool_use_behavior in agents_sdk to control how and when the agent uses tools during execution.

Key Concepts:
- By default, tool_use_behavior is 'run_llm_again'. The LLM can repeatedly call tools and re-run itself until it decides it has enough information to answer the user's query.
- 'stop_on_first_tool': The agent will call tools as needed, but the output of the FIRST tool called will be returned as the final output. The LLM will not receive the tool output for further reasoning.
- StopAtTools(stop_at_tool_names=[...]): You can specify a list of tool names. If any of these tools are called, their output will be returned as the final output, and the LLM will not continue reasoning with the tool output.
- These settings allow you to control the agent's reasoning loop and how tool outputs are handled.

# tool_use_behavior options:
# - Default ('run_llm_again'): LLM can call tools multiple times and re-run itself until it decides to stop.
# - 'stop_on_first_tool': The output of the first tool called is returned as the final output; LLM does not continue reasoning.
# - StopAtTools(stop_at_tool_names=[...]): If any tool in the list is called, its output is returned as the final output.
# This allows for fine-grained control over agent/tool interaction and output handling.
# See agent.py documentation for more details on tool_use_behavior.
"""

import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, StopAtTools, function_tool, set_tracing_export_api_key
from dotenv import load_dotenv, find_dotenv

_: bool = load_dotenv(find_dotenv())

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY", ""))
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

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


base_agent = Agent(
    name="pirate_agent",
    instructions="you are a helpful assistant",
    model=model,
    tools=[get_weather, piaic_student_finder],
    # tool_use_behavior="run_llm_again"
    # tool_use_behavior="stop_on_first_tool"
    tool_use_behavior=StopAtTools(stop_at_tool_names=["get_weather"])
)

result = Runner.run_sync(
    base_agent,
    input="what is the weather in Islamabad in celsius? and who is student with roll number 2?",
)

print(result.final_output)

# Scenario-based Questions:
# 1. What happens if you use the default tool_use_behavior and the agent needs to call multiple tools?
# 2. How does 'stop_on_first_tool' change the agent's response compared to the default behavior?
# 3. If you use StopAtTools(stop_at_tool_names=["get_weather"]) and both tools are called, which output is returned?
# 4. How would you design an agent that always returns the output of a specific tool, regardless of other tool calls?
# 5. What are the trade-offs between allowing the LLM to reason with tool outputs vs. returning tool outputs directly?
# 6. How can you use tracing to observe the agent's reasoning loop and tool call sequence?