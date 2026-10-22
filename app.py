"""
AI Bug Report Analyzer (LangGraph + Gemini)

Workflow:
    START -> bug_analysis -> severity_detection -> root_cause -> format_report -> END
"""

import os
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. Create a .env file with GOOGLE_API_KEY=your_key"
    )

# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# --------------------------------------------------------------------------
# Graph State (shared data passed between nodes)
# --------------------------------------------------------------------------
class BugState(TypedDict, total=False):
    bug_description: str          # user input
    title: str                    # Node 1
    affected_feature: str         # Node 1
    summary: str                  # Node 1
    severity: str                 # Node 2
    severity_reason: str          # Node 2
    root_causes: List[str]        # Node 3
    final_report: str             # Node 4


# --------------------------------------------------------------------------
# Structured output schemas (so Gemini returns clean, predictable data)
# --------------------------------------------------------------------------
class BugAnalysis(BaseModel):
    title: str = Field(description="Short, clear bug title (max 12 words)")
    affected_feature: str = Field(description="The feature/module/component affected")
    summary: str = Field(description="2-3 sentence summary of the issue")


class SeverityResult(BaseModel):
    severity: Literal["Critical", "High", "Medium", "Low"]
    reason: str = Field(description="One or two sentences justifying the severity")


class RootCauseResult(BaseModel):
    possible_causes: List[str] = Field(
        description="3 to 5 possible technical root causes, most likely first"
    )


# --------------------------------------------------------------------------
# Node 1: Bug Analysis
# --------------------------------------------------------------------------
def bug_analysis_node(state: BugState) -> BugState:
    structured_llm = llm.with_structured_output(BugAnalysis)
    prompt = f"""You are a senior QA engineer. Analyze the bug description below.
Extract a bug title, the affected feature, and a short issue summary.

Bug description:
{state['bug_description']}
"""
    result = structured_llm.invoke(prompt)
    return {
        "title": result.title,
        "affected_feature": result.affected_feature,
        "summary": result.summary,
    }


# --------------------------------------------------------------------------
# Node 2: Severity Detection
# --------------------------------------------------------------------------
def severity_detection_node(state: BugState) -> BugState:
    structured_llm = llm.with_structured_output(SeverityResult)
    prompt = f"""Classify the severity of this bug as exactly one of:
Critical, High, Medium, Low.

Guidelines:
- Critical: system crash, data loss, security breach, complete outage, no workaround.
- High: major feature broken, affects many users, difficult workaround.
- Medium: feature partially broken, workaround exists.
- Low: cosmetic issue, typo, minor inconvenience.

Title: {state['title']}
Affected feature: {state['affected_feature']}
Summary: {state['summary']}
"""
    result = structured_llm.invoke(prompt)
    return {"severity": result.severity, "severity_reason": result.reason}


# --------------------------------------------------------------------------
# Node 3: Root Cause Suggestion
# --------------------------------------------------------------------------
def root_cause_node(state: BugState) -> BugState:
    structured_llm = llm.with_structured_output(RootCauseResult)
    prompt = f"""You are a senior software engineer. Suggest possible root causes
for this bug. Be specific and technical.

Title: {state['title']}
Affected feature: {state['affected_feature']}
Summary: {state['summary']}
Severity: {state['severity']}
Original description: {state['bug_description']}
"""
    result = structured_llm.invoke(prompt)
    return {"root_causes": result.possible_causes}


# --------------------------------------------------------------------------
# Node 4: Format Report
# --------------------------------------------------------------------------
SEVERITY_ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}


def format_report_node(state: BugState) -> BugState:
    icon = SEVERITY_ICON.get(state["severity"], "")
    causes = "\n".join(f"  {i}. {c}" for i, c in enumerate(state["root_causes"], 1))

    report = f"""
{'=' * 60}
                    BUG REPORT
{'=' * 60}

TITLE
  {state['title']}

AFFECTED FEATURE
  {state['affected_feature']}

SUMMARY
  {state['summary']}

SEVERITY
  {icon} {state['severity']}
  Reason: {state['severity_reason']}

POSSIBLE ROOT CAUSES
{causes}

{'=' * 60}
"""
    return {"final_report": report.strip()}


# --------------------------------------------------------------------------
# Build the graph
# --------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(BugState)

    graph.add_node("bug_analysis", bug_analysis_node)
    graph.add_node("severity_detection", severity_detection_node)
    graph.add_node("root_cause", root_cause_node)
    graph.add_node("format_report", format_report_node)

    graph.add_edge(START, "bug_analysis")
    graph.add_edge("bug_analysis", "severity_detection")
    graph.add_edge("severity_detection", "root_cause")
    graph.add_edge("root_cause", "format_report")
    graph.add_edge("format_report", END)

    return graph.compile()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    app = build_graph()

    print("=== AI Bug Report Analyzer ===")
    print("Describe the bug in natural language (type 'quit' to exit).\n")

    while True:
        bug = input("Bug description: ").strip()
        if bug.lower() in {"quit", "exit", "q"}:
            break
        if not bug:
            print("Please enter a description.\n")
            continue

        try:
            result = app.invoke({"bug_description": bug})
            print("\n" + result["final_report"] + "\n")
        except Exception as e:
            print(f"\nError while generating report: {e}\n")


if __name__ == "__main__":
    main()
