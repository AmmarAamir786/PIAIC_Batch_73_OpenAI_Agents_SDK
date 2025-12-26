import json
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
import requests

# Set up your Gemini API key here 
gemini_api_key = "AIzaSyAZeh8ytSvbQZr3dc4gjJ71SucyuYOKEIw"

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

SERPAPI_URL = "https://serpapi.com/search.json"

def _serpapi_search(query: str, engine: str = "google", num_results: int = 5):
    api_key = "c2d994912db098f72bca0b002c57646ba895b0a67089ffaf351024d1591ac178"
    if not api_key:
        raise RuntimeError("Missing SERPAPI_KEY env var")

    # Ask for a few results; SerpAPI puts them under "organic_results"
    params = {
        "api_key": api_key,
        "engine": engine,   # "google", "bing", etc.
        "q": query,
        "num": max(1, min(num_results, 10)),  # keep it modest on free tier
        "safe": "active",
    }
    r = requests.get(SERPAPI_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Normalize to a compact list the agent can read easily.
    results = []
    for item in (data.get("organic_results") or [])[:num_results]:
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
            "position": item.get("position"),
            "source": engine,
        })
    return {
        "query": query,
        "engine": engine,
        "results": results,
    }

@function_tool("web_search")
def web_search(query: str, engine: str = "google", num_results: int = 5) -> str:
    """
    Run a web search and return a concise JSON with title, link, and snippet.
    - query: user question or keywords
    - engine: 'google' (default) or another SerpAPI engine
    - num_results: 1-10 results to return
    """
    payload = _serpapi_search(query=query, engine=engine, num_results=num_results)
    # Return a short string (LLMs do great at reading JSON text)
    return json.dumps(payload, ensure_ascii=False)


agent = Agent(
    name="realtime-search-agent",
    instructions=(
        "You are a helpful assistant with web search. "
        "When the user asks for current info, call `web_search` first, "
        "then summarize concisely with links."
    ),
    model=model,
    tools=[web_search],
)

if __name__ == "__main__":
    # Demo: the agent decides when to call `web_search`
    user_query = "Latest news about PSL schedule changes this season"
    result = Runner.run_sync(agent, input=user_query)
    print("\n--- AGENT OUTPUT ---\n")
    print(result.final_output)