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

class WeatherAnswer(BaseModel):
    location: str
    temperature_c: float
    summary: str

base_agent = Agent(
    name="pirate_agent",
    instructions="you are a helpful assistant.",
    model=model,
    output_type=WeatherAnswer
)

result = Runner.run_sync(
    base_agent,
    input="what is the weather in Islamabad in celsius?",
)

print(result.final_output)