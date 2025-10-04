"""
This example demonstrates how to enable verbose tracing/logging in the terminal using agents_sdk.

- The key line is enable_verbose_stdout_logging(), which prints detailed logs of agent execution to your terminal.
- This is useful for debugging, monitoring, and understanding agent behavior in real time without using an external dashboard.
- Unlike OpenAI dashboard tracing, this works for any provider and does not require an API key for tracing.

Use Case:
- Ideal for local development, debugging, and learning how agents and tools interact step by step.
"""

import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool,set_tracing_export_api_key

_: bool = load_dotenv(find_dotenv())

# ONLY FOR TRACING
# Enable verbose logging to print detailed agent/tool execution info to the terminal for debugging and tracing
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()

# Setup Gemini API Key
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

@function_tool
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return f"The weather for {city} is sunny."

base_agent = Agent(
    name="WeatherAgent",
    instructions="You are a helpful assistant.",
    model=llm_model,
    tools=[get_weather]
)

response = Runner.run_sync(
    base_agent, "What's the weather in Karachi?"
)

print(response.final_output)