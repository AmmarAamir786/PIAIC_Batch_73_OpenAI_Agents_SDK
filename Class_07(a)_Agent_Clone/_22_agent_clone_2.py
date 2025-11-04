"""
This example continues agent cloning in agents_sdk, focusing on how tools are handled.

Key Concept:
- When you clone an agent, if you do NOT pass a new list of tools, the cloned agent will share the same tool list as the original agent (shallow copy).
- Any changes made to the tools list (e.g., appending a new tool) will affect BOTH the original and cloned agents.
- If you pass a new list of tools in the clone method, the cloned agent will have its own separate tool list (deep copy).
- This is important for avoiding unintended side effects when customizing cloned agents.
"""

import os
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_export_api_key
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

@function_tool("new_tool")
def new_tool(student_roll: int) -> str:
  """
  find the PIAIC student based on the roll number
  """
  data = {1: "Ammar",
          2: "Azeem",
          3: "Zain"}

  return data.get(student_roll, "Not Found") # Azeem

# Create a base agent with specific instructions, model settings, and tools
base_agent = Agent(
    name="pirate_agent",
    instructions="Write like a pirate who loves math",
    model_settings=ModelSettings(
        temperature=1,
        max_tokens=1024,
        top_p=1,
        ),
    model=model,
    tools=[get_weather, piaic_student_finder], # initial tool list
)

# Clone the base agent, changing only the name and instructions
# No new tools list is passed, so the cloned agent shares the same tool list as the base agent
cloned_agent = base_agent.clone(
    name = "robot_agent",
    instructions = "Write like a pirate who writes using numbers",
    # tools argument is omitted, so the tool list is shared
    # if a tools argument were passed here, the cloned agent would have its own separate tool list
)

# Appending a new tool to the cloned agent's tools list
# Because the tool list is shared, this change will also affect the base agent
cloned_agent.tools.append(new_tool)

# Appending a new tool to the base agent's tools list
# Because the tool list is shared, this change will also affect the cloned agent
# base_agent.tools.append(new_tool)

result_base = Runner.run_sync(base_agent, input="what is 2+2?")
result_cloned = Runner.run_sync(cloned_agent, input="what is 2+2?")

print(result_base.final_output)
print(result_cloned.final_output)

# Scenario-based Questions:
# 1. If you append a new tool to the cloned agent's tools list, does it affect the base agent? Why?
# 2. How would you ensure that the cloned agent has its own independent tool list?
# 3. What are the risks of sharing mutable objects (like lists) between agents when cloning?
# 4. If you modify the base agent's tools after cloning, what happens to the cloned agent's tools?
# 5. How can you use tracing to verify which tools are available to each agent during execution?