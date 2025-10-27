"""
This example demonstrates the use of max_turns in agents_sdk to control the maximum number of reasoning steps (turns/llm calls) the agent can take during execution.

Key Concepts:
- max_turns limits how many times the agent (LLM) can reason and respond before returning a final output.
- If the agent exceeds max_turns, a MaxTurnsExceeded exception is raised and crashes the program.
- This is useful for preventing infinite loops, runaway reasoning, or excessive API usage.
- If max_turns is not specified, the default value is 10 (see agents_sdk source code).
- You can handle MaxTurnsExceeded exceptions to provide a custom error message or fallback behavior.
- If you do not handle the exception, the program will terminate with an error.
- Use max_turns to control cost, latency, and agent behavior in production systems.
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
    max_turns=1 # Set max_turns to 1 to limit the agent to a single llm call.
)

print(result.final_output)

"""
Scenario-based Questions:
1. Why do we need max_turns in agent execution?
2. What happens if you do not specify max_turns? 
3. What is the default value?
4. How can you find the default value of max_turns in the source code?
5. What happens if the agent exceeds max_turns and the MaxTurnsExceeded exception is raised?
6. Can you handle MaxTurnsExceeded exceptions? How would you do it?
7. How does max_turns help prevent infinite loops or excessive API usage?
8. In what scenarios would you want to set max_turns to a low value vs. a high value?
9. How does max_turns interact with tool calling and agent turns?
10. Can you log or trace the number of turns taken by the agent for debugging or monitoring?
"""