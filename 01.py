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

base_agent = Agent(
    name="pirate_agent",
    instructions="Write like a pirate who loves math",
    model_settings=ModelSettings(
        temperature=1,
        max_tokens=1024,
        top_p=1,
        ),
    model=model,
    tools=[get_weather, piaic_student_finder],
)

cloned_agent = base_agent.clone(
    name = "robot_agent",
    instructions = "Write like a pirate who writes using numbers",
)

cloned_agent.tools.append(new_tool)

result = Runner.run_sync(
    base_agent,
    input="what is 2+2?",
)

print(result.final_output)