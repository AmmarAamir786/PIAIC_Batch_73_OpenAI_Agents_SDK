import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool,set_tracing_export_api_key

_: bool = load_dotenv(find_dotenv())

# ONLY FOR TRACING
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()

gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Which LLM Model?
model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)


# Define individual translation agents.
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An English to Spanish translator",
    model=model
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An English to French translator",
    model=model
)

italian_agent = Agent(
    name="italian_agent",
    instructions="You translate the user's message to Italian",
    handoff_description="An English to Italian translator",
    model=model
)

# Compose an orchestrator agent that uses the above agents as tools.
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools provided to translate messages. "
        "If multiple translations are requested, call the relevant tools in order. "
        "Never translate by yourself; always use the provided tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="Translate the user's message to Italian",
        ),
    ],
    model=model
)


async def main():
    
    orchestrator_result = await Runner.run(
        orchestrator_agent,
        input="Translate 'Good morning, have a nice day!' to Spanish",
    )
    print("Orchestrator Result:", orchestrator_result.final_output)

if __name__ == "__main__":
    asyncio.run(main())