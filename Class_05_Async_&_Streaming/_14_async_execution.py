"""
async vs sync

This example demonstrates how to run an agent asynchronously using asyncio and the async Runner.run method.

- In the previous files, we used Runner.run_sync for synchronous execution.
- Here, we define an async main() function and use await Runner.run for asynchronous execution.
- asyncio.run(main()) is used to start the async event loop and run the main coroutine.

Key Differences:
- Synchronous (run_sync): Blocks the main thread until the agent finishes. Simpler for quick scripts or when only one agent/task is needed.
- Asynchronous (run): Allows you to run multiple agents or tasks concurrently, improving efficiency for I/O-bound or multi-agent scenarios.

When to use which:
- Use sync for simple, single-agent, or blocking tasks.
- Use async when you need to run multiple agents, handle I/O, or integrate with other async code.
"""

import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

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

# 3. Set up the agent to use the model
agent = Agent(
    name="agent",
    instructions="You are a helpful assistant",
    model=model,
)

# 4. Set up the runner to use the agent
async def main(): # step 1: create async main function
    result = await Runner.run( # step 2: use await with Runner.run
        agent,
        input="write an essay on programming in 500 words",
    )

    # Print the agent's final output
    print(result.final_output)
    
    
if __name__ == "__main__":
    asyncio.run(main()) # step 3: use asyncio.run to start the event loop and run main()

# Scenario-based questions:
# 1. Why would you need to use async execution for agents instead of sync?
# 2. What is the role of asyncio in this code, and how does it manage concurrency?
# 3. If you want to run multiple agents in parallel, how would you modify this code?
# 4. In what situations would sync code be preferable over async, and vice versa?
# 5. How would you integrate agent runs with other async I/O operations (e.g., web requests, file I/O)?