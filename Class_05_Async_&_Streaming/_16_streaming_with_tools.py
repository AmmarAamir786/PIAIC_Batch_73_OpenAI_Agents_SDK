"""
This example demonstrates how to stream an agent's output while using tools in agents_sdk.
- The agent is set up with a tool (how_many_jokes) that returns a number.
- The agent is instructed to call the tool first, then tell that many jokes.
- Runner.run_streamed is used to stream the agent's output and tool calls in real time.
- The code prints updates as the agent runs, including tool calls, tool outputs, and message outputs.

Key Points:
- Streaming allows you to see intermediate steps and outputs as the agent works.
- Tools can be called during streaming, and their outputs are shown incrementally.
- This is useful for interactive applications, debugging, or understanding agent reasoning.
"""

import asyncio
from agents import Agent, ItemHelpers, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool

# Set up your Gemini API key here 
gemini_api_key = ""

# 1. Set up the provider to use the Gemini API Key
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Set up the model to use the provider
model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

@function_tool
def how_many_jokes() -> int:
    return 10

# 3. Set up the agent to use the model
agent = Agent(
    name="agent",
    instructions="You are a helpful assistant",
    model=model,
)

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
    
    
# Scenario-based questions:
# 1. What are the benefits of streaming agent output when tools are involved?
# 2. How does the agent decide when to call the tool, and how is the tool output integrated into the response?
# 3. What would happen if the tool returned a different number each time? (not hardcoded to 10)
# 4. What are the challenges in error handling when streaming tool calls and agent outputs?
# 5. How does this approach compare to waiting for the full output before displaying anything?