import os
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_export_api_key
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

base_agent = Agent(
    name="agent",
    instructions="Write like a pirate",
    model_settings=ModelSettings(temperature=0.7),
    model=model,
)

result = Runner.run_sync(
    base_agent,
    input="what is 2+2?",
)

print(result.final_output)