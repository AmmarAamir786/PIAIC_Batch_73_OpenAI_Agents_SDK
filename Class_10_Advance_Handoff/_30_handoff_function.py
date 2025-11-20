import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, enable_verbose_stdout_logging, function_tool, handoff,set_tracing_export_api_key

_: bool = load_dotenv(find_dotenv())

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY", ""))

gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An english to spanish translator",
    model=model
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An english to french translator",
    model=model
)

urdu_agent = Agent(
    name="urdu_agent",
    instructions="You translate the user's message to roman Urdu",
    handoff_description="An english to Urdu translator",
    model=model
)


triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's query",
    # handoffs=[urdu_agent, spanish_agent, french_agent],
    handoffs=[handoff(agent=urdu_agent), 
              handoff(agent=spanish_agent), 
              handoff(agent=french_agent)],
    model=model,
)

async def main():
    result = await Runner.run(
        triage_agent,
        input="please translate this to spanish: 'Hello, how are you?'",
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())