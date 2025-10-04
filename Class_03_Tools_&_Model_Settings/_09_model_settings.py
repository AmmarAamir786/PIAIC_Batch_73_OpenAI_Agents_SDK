"""
This example demonstrates how to configure model settings for an agent in agents_sdk.
Model settings allow you to control the behavior of the language model, such as creativity, response length, and sampling.
"""

from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool

# Set up your Gemini API key here 
gemini_api_key = ""

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

agent = Agent(
    name="agent",
    instructions="You are a helpful assistant.",
    model=model,
    model_settings=ModelSettings(
        temperature=0,      # Controls randomness (creativity): 0 = deterministic, higher = more creative
        max_tokens=1024,    # Controls length: Maximum number of tokens in the response
        top_p=1,            # Controls diversity via nucleus sampling (1 = no filtering)
        # frequency_penalty=0, # Penalizes repeated tokens (uncomment to use)
        # presence_penalty=0  # Penalizes new topic introduction (uncomment to use)
    )
)

result = Runner.run_sync(
    agent,
    input="Generate a story on the importance of AI in education",
)
print(result.final_output)

# Questions:
# 1. Does every model accept all the model settings that can be configured using agents_sdk?
# 2. What happens if you set temperature to a very high value?
# 3. If the agent's response is too long, how can you limit it?
# 4. If you want the model to avoid repeating itself, which setting would you adjust?