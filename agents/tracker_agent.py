import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agno.agent import Agent
from agno.tools import tool

from agents.calculator_agent import safe_calculate as _safe_calculate
from agents.web_search_agent import web_search as _web_search
from agents.rag_agent import document_lookup
from llm_client import get_synthesis_model


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely using asteval.
    Input must be a clean math expression like '0.15 * 240' or 'sqrt(256)'.
    Returns the expression and its result.
    """
    return _safe_calculate(expression)


@tool
def search_web(query: str) -> str:
    """
    Search the web for current information using Tavily API.
    Input must be a clean factual search query.
    Returns top 3 results with titles, URLs, and content snippets.
    """
    return _web_search(query)


tracker_agent = Agent(
    name="Tracker Agent",
    model=get_synthesis_model(),
    tools=[document_lookup, calculate, search_web],
    debug_mode=True,
    markdown=True,
    description=(
        "Multi-tool orchestration agent that handles queries "
        "requiring more than one tool to answer completely."
    ),
    instructions=[
        "You are a multi-tool orchestration agent.",
        "You handle queries that require more than one tool to answer fully.",
        "You have access to three tools: document_lookup, calculate, search_web.",
        "",
        "Follow these steps for EVERY query:",
        "Step 1: Read the query carefully and identify ALL distinct sub-tasks.",
        "Step 2: State explicitly which tools you will use and in what order.",
        "Step 3: Execute the FIRST tool call and note the result.",
        "Step 4: Execute the SECOND tool call and note the result.",
        "Step 5: If there are more sub-tasks, continue until all are done.",
        "Step 6: Combine ALL tool results into one unified coherent answer.",
        "Step 7: Cite sources for every part of your answer:",
        "   - For document_lookup: cite chunk number and source file",
        "   - For calculate: show expression and result",
        "   - For search_web: show source URLs",
        "",
        "STRICT RULES:",
        "- NEVER answer any sub-task from memory",
        "- ALWAYS use the correct tool for each sub-task",
        "- NEVER skip a sub-task",
        "- Show which tool you are calling and why at each step",
        "- Do not proceed to synthesis until ALL tool calls are complete",
        "- Each tool call must be visible in your response",
    ],
)
