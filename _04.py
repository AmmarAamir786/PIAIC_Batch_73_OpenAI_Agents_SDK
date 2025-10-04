import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool,set_tracing_export_api_key

_: bool = load_dotenv(find_dotenv())

# # ONLY FOR TRACING
# from agents import enable_verbose_stdout_logging
# enable_verbose_stdout_logging()

gemini_api_key: str = "AIzaSyB-sZ2CyMhJBHEoIGDUgkqeNOFkGSvZyJ8"

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Which LLM Model?
model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

urdu_agent = Agent(
    name="urdu_agent",
    instructions="You translate the user's message to roman Urdu",
    handoff_description="An english to Urdu translator",
    model=model
)

#think of this as an orchestrator or supervisor agent
#It chooses which agent to send data to based on the user's query
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's query",
    handoffs=[urdu_agent],
)

async def main():
    result = await Runner.run(
        triage_agent,
        input="please translate this to urdu: 'Hello, how are you?'",
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())