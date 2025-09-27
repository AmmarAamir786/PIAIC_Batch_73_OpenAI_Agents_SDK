import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

# Set up your Gemini API key here 
gemini_api_key = "AIzaSyAZeh8ytSvbQZr3dc4gjJ71SucyuYOKEIw"

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
    instructions="You are a joker. you write jokes",
    model=model,
)

# 4. Set up the runner to use the agent and generate a joke
async def main():
    result = await Runner.run(
        agent,
        input="write a joke about programming",
    )

    # Print the agent's final output
    print(result.final_output)
    
    
if __name__ == "__main__":
    asyncio.run(main())