"""
Configuring LLM provider at the run level using RunConfig

This example demonstrates configuring the LLM provider at the run level using RunConfig.
Both agents use the same provider and model, which are specified in the run configuration.
This approach allows you to dynamically set providers and models for each run, offering maximum flexibility.

Note:
- The provider and model are not set at the agent, but are passed in RunConfig during execution.
"""

from agents import Agent, RunConfig, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

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

config = RunConfig(
    model=model_1,
    model_provider=provider,
    tracing_disabled=True # Disable tracing for cleaner output. 
)

agent_1 = Agent(
    name="joker_agent",
    instructions="You are a joker. you write jokes",
)

agent_2 = Agent(
    name="maths_agent",
    instructions="You are a maths expert. you solve maths problems",
)

# Runner uses run_config to specify provider and model for this run
result_1 = Runner.run_sync(
    agent_1,
    input="write a joke about programming",
    run_config=config # run_config sets provider and model at run level
)

result_2 = Runner.run_sync(
    agent_2,
    input="what is 2 + 2?",
    run_config=config # run_config sets provider and model at run level
)

print(result_1.final_output)
print(result_2.final_output)

# Questions:
# 1. In the above code, agent_1 and agent_2 are using which models?
#    - Both agents use 'gemini-2.0-flash' as specified in model_1 via run_config.
#
# 2. If we add a new model variable called model_2 and give it another model "gemini-2.5-flash" and pass it in agent_2,
#    then in this case which agent will use which model?
#    - If both runners use the same run_config, both agents will use the model specified in that run_config (e.g., model_1 or model_2).
#      The model set in run_config overrides any model set elsewhere for that run.