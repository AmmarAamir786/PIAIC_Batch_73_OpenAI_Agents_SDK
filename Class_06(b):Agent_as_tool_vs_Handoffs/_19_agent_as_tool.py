"""
This example demonstrates how to use other agents as tools in agents_sdk.

Key Concept:
- Agents can be wrapped and exposed as tools using the as_tool() method.
- This allows you to compose complex workflows where an "orchestrator" agent can delegate tasks to specialized sub-agents (e.g., translators for different languages).
- Each sub-agent is called as a tool, receives the input, and returns its response to the parent/orchestrator agent.

Important Note:
- No previous conversation history, data, or local context is passed from the parent agent to the sub-agent/tool.
- Each sub-agent receives only the input provided by the parent at the time of the tool call, processes it independently, and returns its output.
- This ensures modularity and separation of concerns, but also means sub-agents do not have access to the parent agent's state or context.

Use Cases:
- Multi-step workflows (e.g., translation, summarization, data processing)
- Modular agent design for complex applications
- Delegating specialized tasks to dedicated agents

How it works in this code:
- Three translation agents (Spanish, French, Italian) are created.
- An orchestrator agent is set up with these agents as tools using as_tool().
- The orchestrator agent receives a user input and decides which translation tool(s) to call.
- The sub-agent (e.g., spanish_agent) is called as a tool, processes the input, and returns the translation to the orchestrator.
- The orchestrator then returns the final output to the user.

Limitations: (verify this using tracing)
- Sub-agents do not share memory, context, or conversation history with the parent agent or with each other.
- Each tool call is stateless and isolated.
"""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool,set_tracing_export_api_key, enable_verbose_stdout_logging

_: bool = load_dotenv(find_dotenv())

# Tracing setup - both terminal and OpenAI dashboard (can use one or both)
enable_verbose_stdout_logging() # Local terminal tracing: prints detailed logs to your terminal
set_tracing_export_api_key(os.getenv("OPENAI_API_KEY", "")) # OpenAI dashboard tracing: enables tracing in your OpenAI dashboard

gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# Each sub-agent needs clear instructions for its specific task (e.g., translation to a target language)
# handoff_description provides a summary of the agent's purpose, which is shown to the parent/orchestrator agent when selecting tools
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish", # guides the sub-agent's behavior
    handoff_description="An English to Spanish translator", # describes the tool for the orchestrator
    model=model
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An English to French translator",
    model=model
)

italian_agent = Agent(
    name="italian_agent",
    instructions="You translate the user's message to Italian",
    handoff_description="An English to Italian translator",
    model=model
)

# When adding sub-agents as tools, you must provide a unique tool_name and a tool_description
# tool_name: how the orchestrator refers to this tool
# tool_description: helps the orchestrator decide which tool to call for a given task
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools provided to translate messages. "
        "If multiple translations are requested, call the relevant tools in order. "
        "Never translate by yourself; always use the provided tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish", # unique identifier for the tool
            tool_description="Translate the user's message to Spanish", # description for tool selection
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="Translate the user's message to Italian",
        ),
    ],
    model=model
)


async def main():

    orchestrator_result = await Runner.run(
        orchestrator_agent,
        input="Translate 'Good morning, have a nice day!' to Spanish",
    )
    print("Orchestrator Result:", orchestrator_result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
    

# Scenario-based Questions:
# 1. For a single translation request, how many LLM calls are made during one execution? Can you verify this using tracing?
# 2. If you request translations to multiple languages in a single input, how does the orchestrator decide which sub-agents to call, and how many LLM/tool calls are made?
# 3. What would happen if you changed the orchestrator's instructions to allow it to translate by itself instead of always using the tools?
# 4. How can you use tracing (terminal or OpenAI dashboard) to verify the flow of tool calls and agent interactions?