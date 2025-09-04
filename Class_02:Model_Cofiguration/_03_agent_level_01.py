"""
How to configure LLM Providers at different levels (Global, Run and Agent)

There are three ways to configure LLM providers in agents_sdk:
1. At the agent level (demonstrated in this file)
2. At the model level
3. At the global level

This file shows how to configure the provider at the agent level.
It uses the same code as the previous file, with no changes in logic.

Note:
- Configuring at the agent level allows you to easily switch providers for individual agents,
  making your code more flexible and modular.
"""

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

# Set up your Gemini API key here 
gemini_api_key = ""

# 1. Set up the provider to use the Gemini API Key
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Set up the model to use the provider
 # This is where we set up the model provider. By passing the provider here, you can easily swap out providers for different agents or models, increasing flexibility and modularity.
model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

# 3. Set up the agent to use the model
agent = Agent(
    name="agent",
    instructions="You are a joker. you write jokes",
    model=model, # At this point we are configuring the model at AGENT level.
)

# 4. Set up the runner to use the agent and generate a joke
result = Runner.run_sync(
    agent,
    input="write a joke about programming",
)

# Print the agent's final output
print(result.final_output)