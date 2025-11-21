import asyncio
from datetime import datetime
import os
import time
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from agents import Agent, AgentHooks, RunContextWrapper, RunHooks, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, TContext, TResponseInputItem,set_tracing_export_api_key

_: bool = load_dotenv(find_dotenv())

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY", ""))

gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Create a custom hook class for our agent
class MyRunHooks(RunHooks):
    async def on_llm_start(self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        system_prompt: Optional[str],
        input_items: list[TResponseInputItem],
    ) -> None:
        print(f"LLM is starting for agent: {agent.name}")
        
    async def on_llm_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        response,
    ) -> None:
        print(f"LLM has ended for agent: {agent.name}")
        
    async def on_agent_start(self, context: RunContextWrapper[TContext], agent: Agent) -> None:
        print(f"Agent {agent.name} is starting.")
    
    async def on_agent_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent,
        output: any,
    ) -> None:
        print(f"Agent {agent.name} has ended with output: {output}")
        
    async def on_handoff(self,
        context: RunContextWrapper[TContext],
        from_agent,
        to_agent,
    ) -> None:
        print(f"Handoff from {from_agent.name} to {to_agent.name} is occurring.")

urdu_agent = Agent(
    name="urdu_agent",
    instructions="You translate the user's message to roman Urdu",
    handoff_description="An english to Urdu translator",
    model=model,
)
       
customer_support_agent = Agent(  
    name="Customer Support Specialist",
    instructions="You determine which agent to use based on the user's query",
    model=model,
    handoffs=[urdu_agent],
)

async def main():

    result = await Runner.run(
        customer_support_agent,
        input="please translate this to urdu: 'Hello, how are you?'",
        hooks=MyRunHooks()
    )
    print(result.final_output)
if __name__ == "__main__":
    asyncio.run(main())