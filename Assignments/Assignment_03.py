"""
# 🧠 Assignment 03 — Understanding Agent-as-Tool vs Agent Handoffs

## 🎯 Objective

To **implement, observe, and compare** two agentic design patterns — **Agent-as-Tool** and **Agent Handoffs** — and understand **when and why** each should be used in real-world applications.

By the end of this assignment, you should:

- Be able to **build modular agentic systems** using both patterns.  
- Understand **how context and state are shared (or isolated)**.  
- Know **which approach suits your project use case**.

---

## 🧩 Part 1: Agent-as-Tool System

### 🔍 Concept Recap

In this pattern, a **parent (orchestrator) agent** delegates specific subtasks to **specialized sub-agents**, treating them as tools.  
Each sub-agent executes its task **independently** and returns results back to the parent.  
The sub-agents **do not share memory or conversation history** with the parent or with each other.

---

### 🧱 Task

Build an **Orchestrator Agent** that uses at least **three sub-agents** as tools.  
Each sub-agent should handle a **different content-processing task**.

#### Example tasks (or you can create your own sub-agents):
- **Summarizer Agent**: Summarizes a given article.  
- **Quiz Agent**: Generates quiz questions from a paragraph.  
- **Keyword Agent**: Extracts key terms or phrases.  

---

### ⚙️ Implementation Steps

1. Create a separate `Agent` for each sub-task (summarizer, quiz generator, keyword extractor).  
2. Wrap each sub-agent using `.as_tool(tool_name, tool_description)` and attach them to the orchestrator.  
3. The orchestrator agent should:
   - Identify which tool(s) to call based on the user query.  
   - Combine outputs if multiple tools are used.  
   - Return the final structured response.  
4. Run test prompts such as:
   - `"Summarize this paragraph and extract keywords."`  
   - `"Generate 3 quiz questions from this article."`  
   - `"Summarize and generate questions from this text."`  

---

### 🧠 Observe & Note

- How many LLM calls are made for one query?  
- Is context (previous message) shared between tools?  
- Can the sub-agents remember earlier results or state?  

---

## 🔄 Part 2: Agent Handoff System

### 🔍 Concept Recap

In a **handoff**, a **router/triage agent** receives the user input, decides which specialized agent should handle it, and **hands off the full conversation and context**.  
The sub-agent then continues as the main responder with full access to **history and memory**.

---

### 🧱 Task

Build a **Triage (Router) Agent** that decides between at least **three specialized agents** — for example:
- **Summarizer Agent**  
- **Quiz Agent**  
- **Keyword Agent**

---

### ⚙️ Implementation Steps

1. Create three specialized agents (similar to Part 1).  
2. Create a router agent and attach the others under  
   `handoffs=[summarizer, quiz, keyword]`.  
3. The router should analyze the input and **delegate the full query** to the correct sub-agent using a handoff.  
4. Test with prompts such as:
   - `"Please summarize the following article..."`  
   - `"Make quiz questions from this paragraph..."`  
   - `"List the key terms in this passage..."`  

---

### 🧠 Observe & Note

- How many LLM calls occur during a single handoff?  
- Does the sub-agent have access to full previous context?  
- How does tracing show flow between router and sub-agents?  

---

## 🧩 Discussion Questions

- In which pattern is it easier to **combine outputs** from multiple agents?  
- Which approach is better when **conversation continuity** is important?  
- What happens if you **nest handoffs** (handoff → handoff)?  
- When building a large system (like a **customer support bot**), which combination of both patterns would make sense?  

---

## 💡 Bonus Challenge

### Design a Hybrid System

- A **router agent (handoff-based)** decides the task category.  
- Inside each specialized agent, sub-tasks are performed using **as_tool()** for modularity.  

#### 🎯 Example:
A router delegates to a “**Research Agent**” via handoff,  
which then uses a **Summarizer Tool** and **Citation Generator Tool** internally.

"""