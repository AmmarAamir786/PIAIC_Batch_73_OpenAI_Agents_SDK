"""
This is another example of configuring LLM providers at the agent level.

This example demonstrates agent-level provider configuration in agents_sdk.
Here, two agents are created, each using a different Gemini model:
- joker_agent uses 'gemini-2.0-flash'
- maths_agent uses 'gemini-2.5-flash'

This approach allows you to assign different models to different agents, providing flexibility for specialized tasks.
"""

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

# Set up your Gemini API key here 
gemini_api_key = ""

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model_1 = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

model_2 = OpenAIChatCompletionsModel(
    model='gemini-2.5-flash',
    openai_client=provider,
)

agent_1 = Agent(
    name="joker_agent",
    instructions="You are a joker. you write jokes",
    model=model_1, # The joker agent uses the gemini-2.0-flash model
)

agent_2 = Agent(
    name="maths_agent",
    instructions="You are a maths expert. you solve maths problems",
    model=model_2, # The maths agent uses the gemini-2.5-flash model
)

result_1 = Runner.run_sync(
    agent_1,
    input="write a joke about programming",
)

result_2 = Runner.run_sync(
    agent_2,
    input="what is 2 + 2?",
)

print(result_1.final_output)
print(result_2.final_output)