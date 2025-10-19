"""
This example demonstrates agent cloning in agents_sdk.

Key Concept:
- Agent cloning allows you to create a new agent by copying the configuration, model, and settings from an existing agent, while optionally changing properties like name or instructions.
- All model settings, tools, and provider information are copied to the new agent unless overridden.
- This is useful for creating similar agents with slight behavioral differences (e.g., different writing styles, instructions, or names).
"""

import os
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_export_api_key
from dotenv import load_dotenv, find_dotenv

_: bool = load_dotenv(find_dotenv())

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY", ""))
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

# Create a base agent with specific instructions and model settings
base_agent = Agent(
    name="pirate_agent",
    instructions="Write like a pirate",
    model_settings=ModelSettings(
        temperature=1,
        max_tokens=1024,
        top_p=1,
        ),
    model=model,
)

# Clone the base agent, changing only the name and instructions
# All other settings (model, model_settings, tools, etc.) are copied from the base agent
cloned_agent = base_agent.clone(
    name = "robot_agent", # new name for the cloned agent
    instructions = "Write like a robot", # new instructions for the cloned agent
)

result_base = Runner.run_sync(base_agent, input="what is 2+2?")
result_cloned = Runner.run_sync(cloned_agent, input="what is 2+2?")

print(result_base.final_output)
print(result_cloned.final_output)