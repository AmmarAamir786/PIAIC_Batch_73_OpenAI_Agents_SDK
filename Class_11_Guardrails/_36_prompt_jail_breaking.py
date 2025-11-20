import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,set_tracing_export_api_key

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

customer_support_agent = Agent(  
    name="Customer Support Specialist",
    # applying the guardrail to this agent in system instructions
    instructions="You are a helpful customer support agent for our software company. Only respond to questions related to our software products.",
    model=model,
)

async def main():
    result = await Runner.run(
        customer_support_agent,
        input="tell me about your software and what is 2+2",
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())