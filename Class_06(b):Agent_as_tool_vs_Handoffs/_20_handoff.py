"""
This example demonstrates the use of agent handoffs in agents_sdk.

Key Concept:
- Handoffs allow an agent (the router or triage agent) to delegate the entire query and context to a more specialized sub-agent.
- Unlike agent-as-tool, in a handoff the full conversation history, input, and context are passed to the sub-agent, which then takes over the conversation.
- The triage agent acts as a supervisor or router, deciding which specialized agent should handle the user's request based on the input.

How it works in this code:
- Three specialized agents (Spanish, French, Urdu translators) are created.
- The triage agent is set up with these agents in its handoffs list.
- When a user query is received, the triage agent analyzes the input and routes the request to the appropriate sub-agent.
- The selected sub-agent receives all the data and context, processes the request, and returns the response as final output.

Key Difference from agent-as-tool:
- In handoffs, the sub-agent receives the full context and conversation history, enabling more seamless and stateful delegation.
- In agent-as-tool, only the input for the tool call is passed, and the sub-agent does not have access to the parent agent's context or history.

Use Cases:
- Multi-lingual support, where a router agent delegates to language-specific agents.
- Complex workflows where different agents handle different stages or types of queries.
- Scenarios requiring stateful delegation and continuity of conversation.

Limitations:
- Proper design is needed to avoid confusion or loss of context in deeply nested handoffs.
"""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, enable_verbose_stdout_logging, function_tool,set_tracing_export_api_key

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
    model="gemini-2.0-flash",
    openai_client=external_client
)

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An english to spanish translator",
    model=model
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An english to french translator",
    model=model
)

urdu_agent = Agent(
    name="urdu_agent",
    instructions="You translate the user's message to roman Urdu",
    handoff_description="An english to Urdu translator",
    model=model
)

# Think of this as a router or supervisor agent
# It chooses which agent to send data to based on the user's query
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's query",
    handoffs=[urdu_agent, spanish_agent, french_agent],
    model=model,
)

# The triage agent uses handoffs to route the user's query and full context to the appropriate specialized agent.
# This enables seamless, stateful delegation for complex workflows.

async def main():
    result = await Runner.run(
        triage_agent,
        input="please translate this to urdu: 'Hello, how are you?'",
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
    
# Scenario-based Questions:
# 1. For a single handoff, how many LLM calls are made and how can you verify this using tracing?
# 2. If the user's query could be handled by multiple agents, how does the triage agent decide which one to hand off to?
# 3. What would happen if you nested handoffs (e.g., a sub-agent also performs a handoff)? How is context managed?
# 4. How does the handoff approach compare to agent-as-tool in terms of context sharing and statefulness?
# 5. What are the risks or challenges of deeply nested or chained handoffs in complex workflows?