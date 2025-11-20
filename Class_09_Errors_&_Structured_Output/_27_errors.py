"""
Errors during tool calling

This example demonstrates how errors raised inside tools are handled in agents_sdk and how you can optionally handle exceptions inside your tools to return more meaningful information to the agent.

Key Concepts:
- The agents SDK captures tool execution errors so the overall agent run does not crash; the agent sees an error response rather than the program terminating.
- You do not always need to catch exceptions inside tools because the SDK provides resilience, but catching exceptions allows you to return clearer, user-friendly messages that help the agent generate better final responses.
- If you expect a tool to fail for certain inputs, handle those cases explicitly (e.g., try/except) and return structured or descriptive error strings.
- Returning an informative error from the tool helps the LLM decide how to proceed (retry, ask the user for different input, or provide a fallback answer).
"""

import os
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, StopAtTools, function_tool, set_tracing_export_api_key
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

@function_tool
def divide(a: int, b: int) -> str:
    """Divides two numbers and returns a string result.

    This tool explicitly handles division errors and returns a user-friendly message
    instead of letting an exception bubble up. Even if we omitted the try/except,
    the agents SDK would capture the error, but returning a clear message leads
    to a better agent response flow.
    """
    try:
        result = a / b
        return str(result)
    except ZeroDivisionError:
        # Return a descriptive message the agent can use to inform the user or ask
        # for alternative input.
        return "Error: You cannot divide by zero. Please provide a different divisor."
    except Exception as e:
        # Catch-all: convert unexpected exceptions into a readable message for the agent.
        return f"Error: {type(e).__name__}: {e}"


base_agent = Agent(
    name="pirate_agent",
    instructions="you are a helpful assistant. Use the tools to answer the questions.",
    model=model,
    tools=[divide],
    model_settings=ModelSettings(tool_choice="required"), # Forcing the agent uses tool
)


result = Runner.run_sync(
    base_agent,
    input="what is 10 divided by 0?",
)

print(result.final_output)

"""
Scenario-based Questions:
1. How does the agents SDK behave when a tool raises an exception during execution?
2. Why might you still want to catch exceptions inside tools even if the SDK prevents crashes?
3. What information should a tool return when it fails to help the agent make a good next decision?
4. What if we set tool_use_behavior="stop_on_first_tool" and the first tool call fails—how does that affect the final output?
"""