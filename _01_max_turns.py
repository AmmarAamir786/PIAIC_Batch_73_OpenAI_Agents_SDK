import os
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, StopAtTools, function_tool, set_tracing_export_api_key
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

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
    instructions="you are a helpful assistant. Use the tools to answer the questions.",
    model=model,
    tools=[get_weather, piaic_student_finder],
)


result = Runner.run_sync(
    base_agent,
    input="what is the weather in Islamabad in celsius?",
    max_turns=1
)

print(result.final_output)