"""
dynamic_instructions_example.py

This example demonstrates how to use dynamic instructions in agents_sdk.
Dynamic instructions allow you to generate the agent's instructions at runtime based on the current local context or other factors.

- The instructions parameter can be a function that receives the current context and agent, and returns a string.
- This enables you to personalize or adapt the agent's behavior for each run (e.g., including the user's name or preferences in the instructions).

Comparison to Local Context:
- Local context is only available to your Python code and tools, not the LLM.
- Dynamic instructions use local context to generate agent/LLM context, which is then sent to the LLM and influences its responses.

Key Point:
- Dynamic instructions bridge the gap between private backend data (local context) and what the LLM sees (agent/LLM context).
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

model = OpenAIChatCompletionsModel(
    model='gemini-2.5-flash',
    openai_client=provider,
)

@dataclass
class UserInfo:  
    name: str
    uid: int
    

def dynamic_instructions(
    context: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."


agent = Agent[UserInfo](
    name = "Helper Agent",
    instructions = dynamic_instructions,
    model = model,
)

async def main():
    
    # Create your context object. In a real world scenerio, this could be fetched from a database or user session.
    user_info = UserInfo(name="John", uid=123)  
    
    result = await Runner.run(
        starting_agent=agent,
        input="What is my name?",
        context=user_info,
    )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
    
    
# Scenario-based questions:
# 1. How do dynamic instructions differ from static instructions in terms of personalization?
# 2. Can you use dynamic instructions to include real-time or session-based data in the LLM's context?
# 3. How does this approach compare to using local context only in tools?
# 4. Are there any risks in exposing too much context to the LLM via dynamic instructions?