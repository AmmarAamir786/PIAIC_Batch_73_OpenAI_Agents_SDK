import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, AgentHooks, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,set_tracing_export_api_key

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
class MyAgentHooks(AgentHooks):
    async def on_start(self, context, agent):
        print(f"🕘 {agent.name} is starting up!")
    
    async def on_llm_start(self, context, agent, system_prompt, input_items):
        print(f"📞 {agent.name} is calling llm with input items:{input_items} ")
    
    async def on_llm_end(self, context, agent, response):
        print(f"🧠✨ {agent.name} got response from llm")
        
    async def on_handoff(self, context, agent, source):
        print(f"🏃‍♂️➡️🏃‍♀️ {agent.name} received work from {source.name}")
    
    async def on_tool_start(self, context, agent, tool):
        print(f"🔨 {agent.name} is using {tool.name}")
    
    async def on_tool_end(self, context, agent, tool, result):
        print(f"✅ {agent.name} finished using {tool.name}")
    
    async def on_end(self, context, agent, output):
        print(f"🎉 {agent.name} completed the task!")

urdu_agent = Agent(
    name="urdu_agent",
    instructions="You translate the user's message to roman Urdu",
    handoff_description="An english to Urdu translator",
    model=model,
    hooks=MyAgentHooks(),
)
       
customer_support_agent = Agent(  
    name="Customer Support Specialist",
    instructions="You determine which agent to use based on the user's query",
    model=model,
    handoffs=[urdu_agent],
    hooks=MyAgentHooks(),
)

async def main():
    result = await Runner.run(
        customer_support_agent,
        input="please translate this to urdu: 'Hello, how are you?'",
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())