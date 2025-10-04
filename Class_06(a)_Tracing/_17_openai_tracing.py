"""
This example demonstrates how to enable tracing in the OpenAI dashboard using agents_sdk.

- The key line is set_tracing_export_api_key(os.getenv("OPENAI_API_KEY", ""))
- You provide your OpenAI API key (no credits required) to enable tracing.
- This does NOT use your OpenAI credits or make paid API calls; it only enables tracing/monitoring in your OpenAI dashboard.
- Place this function at the top of your code before running any agents.
- After running your code, you can view traces and debugging information in the OpenAI dashboard.

Use Case:
- Useful for debugging, monitoring, and understanding agent runs without incurring costs.
"""

import os
from agents import (Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_export_api_key)
# This line imports the `load_dotenv` and `find_dotenv` functions from the `dotenv` package,
# which are used to load environment variables from a `.env` file.
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file if it exists.
_: bool = load_dotenv(find_dotenv())

# Set up tracing for debugging or monitoring by providing the OpenAI API key.
# Uses agents sdk global tracing setup function
# Pass your OpenAI API key here to enable tracing. 
# No credits are needed for this.
# os.getenv is used to fetch the value of the environment variable "OPENAI_API_KEY".
# If the environment variable is not set, it defaults to an empty string.
# Make sure to set the OPENAI_API_KEY in your .env file.
set_tracing_export_api_key(os.getenv("OPENAI_API_KEY", ""))

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

# Now check the trace in your OpenAI dashboard under the "Tracing" section.