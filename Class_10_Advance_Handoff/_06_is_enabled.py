from dataclasses import dataclass
import os
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import Agent, OpenAIChatCompletionsModel, TResponseInputItem, handoff, RunContextWrapper, Runner, set_tracing_export_api_key, trace
from agents.handoffs import HandoffInputData
import asyncio
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

_: bool = load_dotenv(find_dotenv())

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY", ""))

gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

@dataclass
class UserContext:
  user_registered: str


def is_user_registered(context: RunContextWrapper, agent: Agent) -> bool:
    return context.context.user_registered == "true"

# Cardiologist agent
cardiologist_agent = Agent(
    name="Cardiologist",
    handoff_description="Handles cardiac and heart-related patient issues",
    instructions=(
        "You are a cardiologist. You will receive a structured problem description "
        "your job is to handle the patient's issue."
    ),
    model=model,
)

# Receptionist agent with input_filter applied to the handoff
receptionist_agent = Agent(
    name="Receptionist",
    instructions=(
        f"You answer general patient questions. If the patient's issue is heart-related "
        "(e.g., chest pain, palpitations, shortness of breath), escalate to the Cardiologist "
        "by producing the required JSON for the handoff."
    ),
    handoffs=[
        handoff(
            agent=cardiologist_agent,
            is_enabled=is_user_registered,
        )
    ],
    model=model,
)

async def main():

    with trace("receptionist workflow"): 
        convo: list[TResponseInputItem] = []

        print("You are now chatting with an assistant agent. Type 'exit' to end the conversation.")
    
        while True:
            user_input = input("Your Message: ")

            if user_input == "exit":
                print("Goodbye!")
                break

            convo.append({"content": user_input, "role": "user"})
            
            result = await Runner.run(
                receptionist_agent,
                input=convo,
                context=UserContext(user_registered="false"),
            )

            print(f"Agent Response: {result.final_output}")

            convo = result.to_input_list() # Override the whole convo history
            
            # print(convo)

if __name__ == "__main__":
    asyncio.run(main())