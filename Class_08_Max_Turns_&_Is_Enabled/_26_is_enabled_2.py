"""
This example demonstrates the dynamic use of is_enabled in agents_sdk, where a function determines if a tool is available based on the current context.

Key Concepts:
- is_enabled can be a function that receives the run context and agent, returning True/False to control tool visibility dynamically.
- This allows you to restrict tool access based on user roles, permissions, or other runtime conditions.
- In this example, the delete_user_database tool is only enabled for users with the 'admin' role.
- The LLM will only see and use the tool if is_enabled returns True for the current context.
- This approach is useful for implementing admin-only or context-sensitive tools in your agent workflows.
"""

import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, StopAtTools, function_tool, set_tracing_export_api_key, RunContextWrapper
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass

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
  

@dataclass
class UserContext:
  user_role: str
  
# Function that determines if the admin-only tool should be enabled for the current user
# Returns True only if the user's role is 'admin'
def is_user_admin(context: RunContextWrapper, agent: Agent) -> bool:
    return context.context.user_role == "admin"

@function_tool(is_enabled=is_user_admin) # Tool only enabled for admin users
def delete_user_database() -> str:
  """[ADMIN ONLY] Deletes the entire user database."""
  return "Database has been deleted."

# Register the admin-only tool with the agent
base_agent = Agent(
    name="pirate_agent",
    instructions="you are a helpful assistant. Use the tools to answer the questions. If the user is an admin, they can delete the user database.",
    model=model,
    tools=[get_weather, piaic_student_finder, delete_user_database],
)

# Demonstrate running with a context that enables the admin-only tool
admin_result = Runner.run_sync(
    base_agent,
    input="Please delete the user database.",
    context=UserContext(user_role="admin"),
)

print(admin_result.final_output)

"""
Scenario-based Questions:
1. How does using a function for is_enabled differ from a static boolean?
2. What happens if the user's role is not 'admin' when trying to use the admin-only tool?
3. How does the agent/LLM know which tools are available at runtime?
4. Can you use more complex logic in the is_enabled function (e.g., multiple roles, permissions)?
5. What are the risks of not properly restricting sensitive tools with is_enabled?
6. How would you test that the dynamic is_enabled logic works as expected?
7. Can multiple tools use different is_enabled functions for fine-grained control?
8. How does dynamic tool enabling improve security and user experience?
9. What happens if the context is missing or malformed in the is_enabled function?
"""