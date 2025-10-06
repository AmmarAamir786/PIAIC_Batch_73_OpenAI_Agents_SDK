"""
Assignment 02: Tracing and Debugging Agent Workflows

Objective:
- Add tracing code for both OpenAI dashboard tracing and terminal (stdout) tracing to every file and concept you have learned so far.
- The goal is to understand the internal workings, flow, and sequence of events in agent execution, tool calls, handoffs, streaming, and all other scenarios.

Instructions:
1. For every code file and concept (e.g., tools, multiple tools, model settings, tool_choice, streaming, agent-as-tool, handoffs, context, etc.), add:
   a. OpenAI dashboard tracing setup using set_tracing_export_api_key(os.getenv("OPENAI_API_KEY", ""))
   b. Terminal tracing setup using enable_verbose_stdout_logging()
2. Run your code and observe the traces in both your terminal and the OpenAI dashboard.
3. Take notes on what you observe: How many LLM calls are made? When are tools called? How is context passed? What happens in streaming or handoff scenarios?

Bonus:
- Try to identify any unexpected behaviors, inefficiencies, or insights from the traces.
- Just like we learned new things when checking the traces of a tool call, is there anything else that we have missed in our other concepts?
"""