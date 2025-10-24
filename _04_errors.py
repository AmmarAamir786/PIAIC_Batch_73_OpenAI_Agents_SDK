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

@function_tool
def divide(a: int, b: int) -> str:
    """Divides two numbers."""
    return a / b
    # try:
    #     result = a / b
    #     return str(result)
    # except ZeroDivisionError:
    #     return "Error: You cannot divide by zero. Please ask for a different number."


base_agent = Agent(
    name="pirate_agent",
    instructions="you are a helpful assistant. Use the tools to answer the questions.",
    model=model,
    tools=[divide],
    model_settings=ModelSettings(tool_choice="required")
)


result = Runner.run_sync(
    base_agent,
    input="what is 10 divided by 0?",
)

print(result.final_output)