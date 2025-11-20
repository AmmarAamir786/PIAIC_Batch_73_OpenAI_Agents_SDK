import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, RunContextWrapper, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, TResponseInputItem, input_guardrail, output_guardrail,set_tracing_export_api_key
from pydantic import BaseModel

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

# Define what our guardrail should output
class ProductGuardrailOutput(BaseModel):
    is_not_product_related: bool
    reasoning: str

# Create a simple, fast agent to do the checking
input_guardrail_agent = Agent( 
    name="Police",
    instructions="Check if the user is asking you anything unrelated to our software products.",
    model=model,
    output_type=ProductGuardrailOutput,
)

# Create our guardrail function
@input_guardrail
async def product_guardrail( 
    ctx: RunContextWrapper[None], 
    agent: Agent, 
    input: str | list[TResponseInputItem]
):
    # Run our checking agent
    result = await Runner.run(input_guardrail_agent, input, context=ctx.context)
    
    print(result.final_output)
    
    # Return the result with tripwire status
    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=result.final_output.is_not_product_related,
    )
    
    
class MessageOutput(BaseModel): 
    response: str
    
class AIOutput(BaseModel): 
    reasoning: str
    is_not_ai: bool

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the output includes any ai related content.",
    model=model,
    output_type=AIOutput,
)
    
@output_guardrail
async def ai_guardrail(  
    ctx: RunContextWrapper, agent: Agent, output: AIOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response, context=ctx.context)
    print(result.final_output)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_not_ai,
    )

customer_support_agent = Agent(  
    name="Customer Support Specialist",
    instructions="You are a helpful customer support agent for our software company. Only respond to questions related to our software products. Our profuct is ai calculator",
    model=model,
    input_guardrails=[product_guardrail],
    output_guardrails=[ai_guardrail],
    output_type=MessageOutput
)

async def main():
    try:
        result = await Runner.run(
            customer_support_agent,
            input="tell me about your software",
        )
        print(result.final_output)
    except InputGuardrailTripwireTriggered:
        print("Product guardrail tripped")
    except OutputGuardrailTripwireTriggered:
        print("AI guardrail tripped")

if __name__ == "__main__":
    asyncio.run(main())