"""
This example demonstrates two types of context in agents_sdk:

1. Local Context:
   - Local context is any data or dependencies you pass to your agent's run (e.g., user info, loggers, helper functions).
   - It is only available to your Python code (tools, hooks, callbacks) and is never sent to the LLM.
   - You pass it via the context argument to Runner.run(), and tools access it via RunContextWrapper.
   - Example: Storing private user details or dependencies for backend logic.

2. Agent/LLM Context:
   - Agent/LLM context is the information the LLM sees when generating responses (e.g., instructions, system prompts, user input, conversation history).
   - This context is deliberately exposed to the LLM to guide its output.
   - You can embed important context in the agent's instructions or input messages.
   - Example: Including the user's name or current date in the agent's instructions so the LLM can use it in its responses.

Key Difference:
- Local context is internal and never sent to the LLM; it's for backend logic only.
- Agent/LLM context is part of the conversation and is visible to the LLM to influence its responses.
"""

import asyncio
from dataclasses import dataclass
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunContextWrapper, function_tool

# Set up your Gemini API key here 
gemini_api_key = ""

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# You can try both Gemini models below and compare their outputs.
# Uncomment one model at a time to see the difference.

# model = OpenAIChatCompletionsModel(
#     model='gemini-2.0-flash',
#     openai_client=provider,
# )

model = OpenAIChatCompletionsModel(
    model='gemini-2.5-flash',
    openai_client=provider,
)

# To compare outputs, you could run the script twice, each time with a different model uncommented.
# Observe any differences in the agent's responses or behavior.

@dataclass
class UserInfo:  
    name: str
    uid: int

# A tool function that accesses local context via the wrapper.
@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:  
    return f"User {wrapper.context.name} is 47 years old"

async def main():
    # Create your context object
    user_info = UserInfo(name="John", uid=123)  

    # Define an agent that will use the tool above
    agent = Agent[UserInfo](  
        name="Assistant",
        tools=[fetch_user_age],
        model=model,
    )

    # Run the agent, passing in the local context
    result = await Runner.run(  # shifted from using run_sync to run for async compatibility. Will learn the differences later.
        starting_agent=agent,
        input="What is the age of the user?",
        context=user_info,
    )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
    
# To provide agent/LLM context, you can add details directly in the agent's instructions or input message.
# For example, instructions=f"You are helping {user_info.name}." would expose the user's name to the LLM.
# Local context (as shown in this code) is only for your Python code and tools, not the LLM.
    
# Scenario-based questions (Local Context):
# 1. What happens if you pass a different context object to the agent? How does it affect the tool's response?
# 2. Can you pass more complex data (e.g., user preferences, history) in the context?
# 3. How would you modify the tool to return different information based on the context?
# 4. What is the difference between using run_sync and run for agent execution based on the current code?
# 5. Can multiple tools access and use the same context in a single run?

# Scenario-based questions (Agent/LLM Context):
# 1. If you include the user's name in the agent's instructions, how does it affect the LLM's response?
# 2. How would you design instructions to ensure the LLM always greets the user by name?
# 3. What are the risks of exposing sensitive information in the agent/LLM context?