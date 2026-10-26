"""
Web Research Agent
==================
A LangGraph agent that researches a question on the web (with Tavily)
and writes a cited Markdown report (with Claude).

Graph:
    START -> plan_queries -> web_search -> reflect --(need more info?)--> web_search
                                                   +--(enough info)----> write_report -> END
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")


# ---------------------------------------------------------------------------
# 1. State: the "memory" that flows between the nodes of the graph
# ---------------------------------------------------------------------------
class Source(TypedDict):
    title: str
    url: str
    content: str


class ResearchState(TypedDict):
    question: str                                   # the user's question
    queries: list[str]                              # queries for the next search round
    sources: Annotated[list[Source], operator.add]  # all results (appended each round)
    iteration: int                                  # how many search rounds we've done
    max_iterations: int                             # safety limit on search rounds
    report: str                                     # final Markdown answer


# ---------------------------------------------------------------------------
# 2. Structured outputs: we ask Claude to answer in these exact shapes
# ---------------------------------------------------------------------------
class SearchPlan(BaseModel):
    queries: list[str] = Field(
        description="2-4 short, specific web search queries that together cover the question."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="True if the sources are enough to write a complete, well-supported answer."
    )
    missing: str = Field(description="What important information is still missing (empty if none).")
    follow_up_queries: list[str] = Field(
        default_factory=list,
        description="1-3 new search queries to fill the gaps (empty if is_sufficient is true).",
    )


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
def get_llm(model: str | None = None) -> ChatAnthropic:
    return ChatAnthropic(model=model or DEFAULT_MODEL, temperature=0, max_tokens=4096)


def format_sources(sources: list[Source], max_chars: int = 1200) -> str:
    """Number the sources so Claude can cite them as [1], [2], ..."""
    blocks = []
    for i, s in enumerate(sources, start=1):
        blocks.append(f"[{i}] {s['title']}\nURL: {s['url']}\n{s['content'][:max_chars]}")
    return "\n\n".join(blocks)


def dedupe(sources: list[Source]) -> list[Source]:
    seen, unique = set(), []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            unique.append(s)
    return unique


# ---------------------------------------------------------------------------
# 4. Nodes: each node is a function that reads the state and returns updates
# ---------------------------------------------------------------------------
def build_graph(model: str | None = None, results_per_query: int = 4):
    llm = get_llm(model)
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def plan_queries(state: ResearchState) -> dict:
        """Ask Claude to turn the question into a few good search queries."""
        planner = llm.with_structured_output(SearchPlan)
        plan = planner.invoke(
            "You are a research planner. Break the user's question into 2-4 focused "
            "web search queries that together cover everything needed to answer it.\n\n"
            f"Question: {state['question']}"
        )
        return {"queries": plan.queries, "iteration": 0}

    def web_search(state: ResearchState) -> dict:
        """Run every planned query through Tavily and collect the results."""
        found: list[Source] = []
        for query in state["queries"]:
            response = tavily.search(query=query, max_results=results_per_query)
            for r in response.get("results", []):
                found.append({"title": r.get("title", ""), "url": r["url"], "content": r.get("content", "")})
        return {"sources": found, "iteration": state["iteration"] + 1}

    def reflect(state: ResearchState) -> dict:
        """Ask Claude whether we have enough information, or what to search next."""
        critic = llm.with_structured_output(Reflection)
        result = critic.invoke(
            "You are reviewing research notes. Decide whether the sources below are "
            "enough to fully answer the question. If not, propose follow-up queries.\n\n"
            f"Question: {state['question']}\n\nSources:\n{format_sources(dedupe(state['sources']), 500)}"
        )
        queries = [] if result.is_sufficient else result.follow_up_queries
        return {"queries": queries}

    def should_continue(state: ResearchState) -> str:
        """Conditional edge: loop back to search, or move on to writing."""
        if state["queries"] and state["iteration"] < state["max_iterations"]:
            return "web_search"
        return "write_report"

    def write_report(state: ResearchState) -> dict:
        """Ask Claude to write the final answer, citing sources by number."""
        sources = dedupe(state["sources"])
        response = llm.invoke(
            "You are an expert research writer. Using ONLY the sources below, write a "
            "clear, well-structured Markdown report that answers the question.\n"
            "Rules:\n"
            "- Start with a 2-3 sentence summary.\n"
            "- Use headings and bullet points where helpful.\n"
            "- Cite facts inline with the source number, e.g. [1] or [2][3].\n"
            "- If sources disagree or information is missing, say so.\n"
            "- Do NOT add a references list; it is added automatically.\n\n"
            f"Question: {state['question']}\n\nSources:\n{format_sources(sources)}"
        )
        references = "\n".join(f"{i}. [{s['title'] or s['url']}]({s['url']})" for i, s in enumerate(sources, 1))
        report = f"{response.content}\n\n## Sources\n{references}"
        return {"report": report}

    # -----------------------------------------------------------------------
    # 5. Wire the nodes together into a graph
    # -----------------------------------------------------------------------
    graph = StateGraph(ResearchState)
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("web_search", web_search)
    graph.add_node("reflect", reflect)
    graph.add_node("write_report", write_report)

    graph.add_edge(START, "plan_queries")
    graph.add_edge("plan_queries", "web_search")
    graph.add_edge("web_search", "reflect")
    graph.add_conditional_edges("reflect", should_continue, ["web_search", "write_report"])
    graph.add_edge("write_report", END)

    return graph.compile()


def initial_state(question: str, max_iterations: int = 2) -> ResearchState:
    return {
        "question": question,
        "queries": [],
        "sources": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "report": "",
    }
