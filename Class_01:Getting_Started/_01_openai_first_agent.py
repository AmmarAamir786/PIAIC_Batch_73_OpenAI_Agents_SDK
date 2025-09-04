"""
create_your_first_agent.py

This code demonstrates how to create your first agent using the agents_sdk.
It sets up a simple assistant agent and runs it synchronously to generate jokes.

Requirements:
- You must have a valid (paid) OpenAI API key to use this code.
  Insert your API key in the 'OPENAI_API_KEY' environment variable below.

Usage:
- Replace the empty string in os.environ["OPENAI_API_KEY"] with your OpenAI API key.
- Run the code to see the agent's output.

Note:
- However please note that the OpenAI API is a paid service.
"""

import os
from agents import Agent, Runner

# Insert your OpenAI API key here. The API key is required and must be paid.
os.environ["OPENAI_API_KEY"] = "" # <-- Replace with your OpenAI API key

# Create an agent with a name and instructions
agent = Agent(name="Assistant", instructions="You are a joker")

# Run the agent synchronously with a prompt
result = Runner.run_sync(
    agent, # The agent to run
    input="Write a joke about programming") # The prompt the user sends to the agent

# Print the agent's final output
print(result.final_output)
