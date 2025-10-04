"""
Configuring LLM provider at the global level using RunConfig

This example demonstrates configuring the LLM provider and model at the global level using set_* functions.
The following global settings are applied:
- set_tracing_disabled(True): disables tracing for all runs.
- set_default_openai_api("chat_completions"): sets the default API type for all agents.
- set_default_openai_client(provider): sets the default provider for all agents.

Agents can then specify only the model name, and the global provider will be used automatically.

This approach is useful when you want all agents to share the same provider and API configuration.
"""

from agents import Agent, Runner, AsyncOpenAI, set_default_openai_api, set_default_openai_client, set_tracing_disabled

# Set up your Gemini API key here 
gemini_api_key = ""

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Global configuration functions:
set_tracing_disabled(True) # Disables tracing globally for cleaner output
set_default_openai_api("chat_completions") # Sets the default API type for all agents
set_default_openai_client(provider) # Sets the default provider for all agents


agent_1 = Agent(
    name="joker_agent",
    instructions="You are a joker. you write jokes",
    model="gemini-2.0-flash", # The joker agent uses the gemini-2.0-flash model
)

agent_2 = Agent(
    name="maths_agent",
    instructions="You are a maths expert. you solve maths problems",
    model="gemini-2.5-flash", # The maths agent uses the gemini-2.5-flash model
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