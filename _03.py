import asyncio
from random import random
from agents import Agent, ItemHelpers, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool,set_tracing_export_api_key,set_trace_processors
import os
from dotenv import load_dotenv


load_dotenv()

# Set up your Gemini API key here
gemini_api_key = os.getenv("GEMINI_API_KEY")

# 1. Set up the provider to use the Gemini API Key
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Set up the model to use the provider
model = OpenAIChatCompletionsModel(
    model='gemini-2.5-flash',
    openai_client=provider,
)

@function_tool
def how_many_jokes() -> int:
    return 10

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))



# 4. Set up the runner to use the agent and generate a joke
async def main():
    agent = Agent(
        name="Joker",
        instructions="First call the `how_many_jokes` tool, then tell that many jokes.",
        tools=[how_many_jokes],
        model=model,   
    )

    result = Runner.run_streamed(
        agent,
        input="Tell me some jokes",
    )
    print("=== Run starting ===")

    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        if event.type == "raw_response_event":
            # print(f"RAW: {event.data}")
            continue
        # When the agent updates, print that
        elif event.type == "agent_updated_stream_event":
            # print(type(event))
            print(f"Agent updated: {event.new_agent.name}")
            continue
        # When items are generated, print them
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                pass

    print("=== Run complete ===")
    
    
    
if __name__ == "__main__":
    asyncio.run(main())