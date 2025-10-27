"""
This example demonstrates the use of is_enabled in agents_sdk to control whether a tool is available to the agent (LLM) at runtime.

Key Concepts:
- is_enabled determines if a tool is visible and usable by the agent during execution.
- It can be a boolean (True/False) for static enabling/disabling, or a callable (function) that takes the run context and agent and returns a boolean for dynamic control.
- Disabled tools are hidden from the LLM and cannot be called during the turn.
- Use is_enabled to restrict tool access based on user role, context, or other conditions.
- This allows for flexible, context-aware tool availability and safer agent behavior.
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

@function_tool(is_enabled=False) # Disable this tool for the agent
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

  return data.get(student_roll, "Not Found")

base_agent = Agent(
    name="pirate_agent",
    instructions="you are a helpful assistant. Use the tools to answer the questions.",
    model=model,
    tools=[get_weather, piaic_student_finder],
)

result = Runner.run_sync(
    base_agent,
    input="what is the weather in Islamabad in celsius?",
)

print(result.final_output)

"""
Scenario-based Questions:
1. What is the purpose of is_enabled in tool definitions?
2. How does setting is_enabled=False affect agent behavior and tool visibility?
3. What happens if a tool is disabled at runtime? Can the LLM see or use it?
4. In what scenarios would you want to restrict tool access using is_enabled?
5. How does is_enabled interact with agent instructions and reasoning?
6. What are the risks of exposing sensitive tools without proper is_enabled checks?
7. How would you test that is_enabled is working as expected in your agent?
"""