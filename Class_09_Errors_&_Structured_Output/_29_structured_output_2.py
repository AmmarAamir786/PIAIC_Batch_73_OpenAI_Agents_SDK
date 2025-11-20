"""
Meeting minutes extraction (structured output example)

This example shows how to use a Pydantic model as the agent's output_type so the
agent returns structured meeting minutes instead of free-form text. Pydantic
models provide validation, parsing, and convenient access to fields, which is
useful when consuming agent outputs programmatically.
"""

import os
from typing import List, Optional
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_export_api_key
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

# Type annotations explained:
# - str: a string value (text).
# - int: an integer value (whole number).
# - List[T]: a list/array where each element is of type T (e.g., List[str] is a list of strings).
# - Optional[T]: shorthand for Union[T, None]; the field can be a value of type T or None (i.e., optional/missing).
# - Default values (e.g., priority: str = "medium"): if the agent doesn't provide the field, the default will be used.
# - Pydantic BaseModel: validates incoming data against the declared types, coerces types when possible, and provides convenient methods like .dict() and .json(). If validation fails, Pydantic raises a ValidationError.

class ActionItem(BaseModel):
    task: str  # task: string - short description of the action to be performed
    assignee: str  # assignee: string - name of the person responsible for the task
    due_date: Optional[str] = None  # due_date: optional string - deadline or target date (ISO or human-readable)
    priority: str = "medium"  # priority: string - priority level (e.g., low, medium, high)

class Decision(BaseModel):
    topic: str  # topic: string - subject or area the decision relates to
    decision: str  # decision: string - the outcome or choice that was made
    rationale: Optional[str] = None  # rationale: optional string - explanation or reasoning for the decision

class MeetingMinutes(BaseModel):
    meeting_title: str  # meeting_title: string - title or name of the meeting
    date: str  # date: string - date of the meeting (ISO format or human-readable)
    attendees: List[str]  # attendees: list[str] - names of participants
    agenda_items: List[str]  # agenda_items: list[str] - agenda topics discussed
    key_decisions: List[Decision]  # key_decisions: list[Decision] - decisions recorded during the meeting
    action_items: List[ActionItem]  # action_items: list[ActionItem] - tasks assigned with assignees and due dates
    next_meeting_date: Optional[str] = None  # next_meeting_date: optional string - scheduled date for the next meeting
    meeting_duration_minutes: int  # meeting_duration_minutes: int - duration of the meeting in minutes

base_agent = Agent(
    name="MeetingSecretary",
    instructions="""Extract structured meeting minutes from meeting transcripts.
    Identify all key decisions, action items, and important details.""",
    output_type=MeetingMinutes, # Use the MeetingMinutes Pydantic model for structured output
    model=model,
)

meeting_transcript = """
Marketing Strategy Meeting - January 15, 2024
Attendees: Sarah (Marketing Manager), John (Product Manager), Lisa (Designer), Mike (Developer)
Duration: 90 minutes

Agenda:
1. Q1 Campaign Review
2. New Product Launch Strategy  
3. Budget Allocation
4. Social Media Strategy

Key Decisions:
- Approved $50K budget for Q1 digital campaigns based on strong ROI data
- Decided to launch new product in March instead of February for better market timing
- Will focus social media efforts on Instagram and TikTok for younger demographics

Action Items:
- Sarah to create campaign timeline by January 20th (high priority)
- John to finalize product features by January 25th
- Lisa to design landing page mockups by January 22nd
- Mike to review technical requirements by January 30th

Next meeting: January 29, 2024
"""

result = Runner.run_sync(
    base_agent,
    input=meeting_transcript,
)

print(result.final_output)