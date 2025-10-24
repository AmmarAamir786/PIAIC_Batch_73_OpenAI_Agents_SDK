import os
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, StopAtTools, function_tool, set_tracing_export_api_key, RunContextWrapper
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
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
  
# This function checks the context provided during the run.
def is_user_admin(context: RunContextWrapper, agent: Agent) -> bool:
    return context.context.user_role == "admin"

@function_tool(is_enabled=is_user_admin)
def delete_user_database() -> str:
  """[ADMIN ONLY] Deletes the entire user database."""
  return "Database has been deleted."

# register the admin-only tool with the agent
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