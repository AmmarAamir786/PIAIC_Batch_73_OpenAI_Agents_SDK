"""
This example demonstrates using the agent's `output_type` to produce structured outputs
instead of plain strings.

What is output_type?
- output_type: type[Any] | AgentOutputSchemaBase | None = None
- If not provided, the output will be `str`.

Why use output_type?
- Structured outputs (Pydantic models, dataclasses, TypedDicts, etc.) make it
  easy to validate, parse, and consume the agent's response programmatically.
- They help the agent produce consistent, machine-readable outputs which is
  especially useful when the output will be further processed by code.

What types can you use?
- Pydantic models (most common): provide automatic validation, parsing, and
  convenient .dict()/.json() methods. They also generate JSON Schemas which
  can be used for strict validation or tooling.
- Dataclasses: lightweight and native to Python; useful when you want simple
  typed containers without extra validation.
- TypedDict: useful for dictionary-like structured output with static typing.

Why Pydantic is commonly used:
- Strong validation of incoming data (types, required fields).
- Easy parsing of raw values into Python types (e.g., parsing numbers, dates).
- Built-in serialization (json, dict) and developer ergonomics (models are
  self-documenting and work well with IDEs).

In this file we use a Pydantic model (WeatherAnswer) as the output_type so the
agent returns structured weather information.
"""

import os
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, StopAtTools, function_tool, set_tracing_export_api_key
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

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

# Pydantic model used as the structured output schema. Pydantic will validate
# and parse the fields returned by the agent. This gives you a reliable Python
# object (with .dict()/.json()) instead of raw text to parse manually.
class WeatherAnswer(BaseModel):
    location: str
    temperature_c: float
    summary: str

base_agent = Agent(
    name="pirate_agent",
    instructions="you are a helpful assistant.",
    model=model,
    output_type=WeatherAnswer
)

result = Runner.run_sync(
    base_agent,
    input="what is the weather in Islamabad in celsius?",
)

print(result.final_output)

"""
Scenario-based Questions:
1. What are the benefits of using a Pydantic model as output_type versus a plain string?
2. When might you prefer a dataclass or TypedDict over Pydantic for output_type?
3. How does output_type help downstream systems that consume agent responses?
4. How do Pydantic models improve developer experience compared to manual parsing?

Advance questions for topics not covered in class:
4. What would you do if the agent returns data that doesn't conform to your schema?
5. How can you make the agent produce stricter JSON that matches your Pydantic schema?
6. When should you use AgentOutputSchema with strict_json_schema=False?
"""