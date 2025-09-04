"""
create your first agent absolutely free using Gemini API

This code demonstrates how to use Gemini as a free alternative to OpenAI for agent-based tasks.
You will need to set up your Gemini API key to use this code.

Steps:
1. Obtain a Gemini API key (free tier available).
2. Insert your Gemini API key in the 'gemini_api_key' variable below.
3. The code configures the provider, model, agent, and runner to generate a response.

Note:
- Gemini API offers a free tier, making it a cost-effective alternative to OpenAI.
- Make sure your API key is valid and has access to the Gemini endpoints.
"""

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

# Set up your Gemini API key here (free tier available)
gemini_api_key = "" # <-- Add your Gemini API key

# 1. Set up the provider to use the Gemini API Key
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
    # This is Gemini API endpoint. Get this from GEMINI documentation. This is where the gemini models are hosted.
    # Every model provider has its own endpoint (base_url)
    # So if you switch to another provider e.g Grok, make sure to change the base_url accordingly.
)

# 2. Set up the model to use the provider
model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash', # Specify the Gemini model you want to use. Make sure the model name is as per Gemini documentation.
    openai_client=provider,
)

# 3. Set up the agent to use the model
agent = Agent(
    name="agent", # name of the agent. You can choose any name.
    instructions="You are a joker. you write jokes", # instructions for the agent. This guides the agent's behavior. 
    model=model,
)

# 4. Set up the runner to use the agent and generate a joke
result = Runner.run_sync(
    agent, # The agent to run
    input="write a joke about programming", # The prompt the user sends to the agent
)

# Print the agent's final output
print(result.final_output)