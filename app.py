"""Streamlit frontend for the CPG/Retail Analytics LangGraph + MCP agent.

Expects your existing agent script to be saved next to this file as `agent.py`
(we only import SYSTEM_PROMPT from it, so the prompt lives in one place).
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import streamlit as st
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent import SYSTEM_PROMPT

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="CPG Analytics Agent", page_icon="📈", layout="wide")


# ---------------------------------------------------------
# Settings (st.secrets -> env var -> default)
# ---------------------------------------------------------
def get_setting(key: str, default: str) -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


OLLAMA_BASE_URL = get_setting("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = get_setting("OLLAMA_MODEL", "qwen2.5:3b")
MCP_SERVER_PATH = str(BASE_DIR / get_setting("MCP_SERVER_PATH", "mcp_servers/pricing.py"))


# ---------------------------------------------------------
# MCP tools (discovered once per server process)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to MCP server...")
def load_tools():
    client = MultiServerMCPClient(
        {
            "price_elasticity": {
                "command": sys.executable,
                "args": [MCP_SERVER_PATH],
                "transport": "stdio",
            }
        }
    )
    return asyncio.run(client.get_tools())


# ---------------------------------------------------------
# LangGraph agent
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(tools):
    model = ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0
    ).bind_tools(tools)

    async def call_model(state: AgentState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    wf = StateGraph(AgentState)
    wf.add_node("agent", call_model)
    wf.add_node("tools", ToolNode(tools))
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", tools_condition)
    wf.add_edge("tools", "agent")
    return wf.compile()


async def run_agent(history: list[BaseMessage], tools, status) -> tuple[str, list[dict]]:
    """Run the graph, writing tool activity into `status`. Returns (answer, tool_log)."""
    graph = build_graph(tools)
    inputs = {"messages": [SystemMessage(content=SYSTEM_PROMPT), *history]}

    answer = ""
    tool_log: list[dict] = []

    async for event in graph.astream(inputs, stream_mode="values"):
        msg = event["messages"][-1]

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                status.write(f"🔧 Calling **{tc['name']}**")
                status.json(tc.get("args", {}))
                tool_log.append({"kind": "call", "name": tc["name"], "data": tc.get("args", {})})

        elif msg.type == "tool":
            status.write(f"📥 Result from **{msg.name}**")
            status.code(str(msg.content)[:3000])
            tool_log.append({"kind": "result", "name": msg.name, "data": str(msg.content)[:3000]})

        elif msg.type == "ai" and msg.content:
            answer = msg.content

    return answer, tool_log


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # display: {"role", "content", "tool_log"}
if "history" not in st.session_state:
    st.session_state.history = []  # LangChain Human/AI messages for context

with st.sidebar:
    st.header("📈 CPG Analytics Agent")
    st.caption("Price elasticity and demand forecasting via MCP tools.")

    try:
        tools = load_tools()
        st.success(f"{len(tools)} tool(s) connected")
        for t in tools:
            st.markdown(f"- `{t.name}`")
        if "_forecast_demand" not in [t.name for t in tools]:
            st.warning("`_forecast_demand` is not exposed by the MCP server.")
    except Exception as e:
        st.error(f"Could not load MCP tools: {e}")
        st.stop()

    st.divider()
    st.caption(f"Model: `{OLLAMA_MODEL}`")
    st.caption(f"Endpoint: `{OLLAMA_BASE_URL}`")

    st.subheader("Try an example")
    examples = [
        "Forecast 6 weeks of demand for BAG SNACKS",
        "Calculate price elasticity for UPC 1111038078",
        "Give me a 12 week forecast for COLD CEREAL",
    ]
    clicked = None
    for ex in examples:
        if st.button(ex, use_container_width=True):
            clicked = ex

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

st.title("CPG / Retail Analytics")

# Replay conversation
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("tool_log"):
            with st.expander("Tool activity"):
                for item in m["tool_log"]:
                    label = "Call" if item["kind"] == "call" else "Result"
                    st.markdown(f"**{label}: {item['name']}**")
                    if item["kind"] == "call":
                        st.json(item["data"])
                    else:
                        st.code(item["data"])

prompt = st.chat_input("Ask about price elasticity or demand forecasts...") or clicked

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "tool_log": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.history.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        status = st.status("Working...", expanded=True)
        try:
            answer, tool_log = asyncio.run(
                run_agent(st.session_state.history, tools, status)
            )
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Failed", state="error")
            answer, tool_log = f"Something went wrong: `{e}`", []

        st.markdown(answer or "_No response returned._")

    st.session_state.history.append(AIMessage(content=answer))
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tool_log": tool_log}
    )