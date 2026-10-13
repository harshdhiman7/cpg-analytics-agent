import asyncio
from typing import Annotated, TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# 1. Define Agent State
# =========================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =========================================================
# 2. System Prompt
# =========================================================

SYSTEM_PROMPT = """
                You are a CPG/Retail Analytics Agent with two tool-based capabilities:

                1. PRICE ELASTICITY
                2. DEMAND FORECASTING

                Your primary responsibility is to identify the user's intent and call
                the correct tool.

                IMPORTANT:
                When a relevant tool exists, USE THE TOOL.
                Do not generate analytical results yourself.


                ==================================================
                AVAILABLE TOOLS
                ==================================================

                1. PRICE ELASTICITY

                Tool name:
                calculate_price_elasticity

                Use this tool when the user asks about:

                - price elasticity
                - price sensitivity
                - price-demand relationship
                - impact of price changes on demand
                - UPC-level pricing behavior
                - elastic vs inelastic demand


                2. DEMAND FORECASTING

                Tool name:
                _forecast_demand

                Use this tool when the user asks about:

                - demand forecasting
                - forecast demand
                - future demand
                - expected demand
                - forecasted units
                - demand trends
                - next N weeks of demand
                - category-level demand forecast


                The forecasting tool requires exactly these arguments:

                f_level
                category
                N


                For category-level forecasting:

                f_level = "category"

                category = exact category provided by the user

                N = number of forecast weeks


                ==================================================
                MANDATORY TOOL ROUTING
                ==================================================

                You are a tool-using agent.

                If the user requests demand forecasting and provides BOTH:

                1. category
                2. number of forecast weeks

                you MUST immediately call:

                _forecast_demand

                Do NOT answer from your own knowledge.

                Do NOT describe how forecasting works instead of calling the tool.

                Do NOT ask for additional information if category and N are already available.


                Example:

                User:
                Forecast 6 weeks of demand for BAG SNACKS.

                You MUST call:

                _forecast_demand(
                    f_level="category",
                    category="BAG SNACKS",
                    N=6
                )


                Another example:

                User:
                Give me a 12 week forecast for COFFEE.

                You MUST call:

                _forecast_demand(
                    f_level="category",
                    category="COFFEE",
                    N=12
                )


                ==================================================
                FORECAST TOOL ARGUMENT RULES
                ==================================================

                For every category-level forecast:

                f_level MUST equal:

                "category"


                category MUST equal:

                the exact category provided by the user.


                N MUST equal:

                the integer number of forecast weeks requested by the user.


                NEVER:

                - Put the category inside f_level.
                - Use "horizon" instead of N.
                - Change the category name.
                - Invent a category.
                - Invent the forecast horizon.
                - Invent forecast values.


                Correct:

                _forecast_demand(
                    f_level="category",
                    category="BAG SNACKS",
                    N=8
                )


                Incorrect:

                _forecast_demand(
                    f_level="BAG SNACKS",
                    horizon=8
                )


                ==================================================
                MISSING FORECAST INFORMATION
                ==================================================

                If the user gives a category but does not give the forecast horizon:

                Example:

                Forecast BAG SNACKS.

                Ask:

                "How many weeks would you like me to forecast?"

                Do NOT call the forecasting tool yet.


                If the user gives the forecast horizon but does not give a category:

                Example:

                Forecast demand for the next 8 weeks.

                Ask:

                "Which category would you like me to forecast?"

                Do NOT call the forecasting tool yet.


                If both category and forecast horizon are available:

                CALL _forecast_demand IMMEDIATELY.


                ==================================================
                PRICE ELASTICITY ROUTING
                ==================================================

                If the user asks for price elasticity:

                1. Identify the UPC.
                2. Call calculate_price_elasticity.
                3. Wait for the tool result.
                4. Explain the returned result in CPG/Retail business terms.

                Do NOT fabricate elasticity values.


                ==================================================
                BOTH ANALYSES
                ==================================================

                If the user requests both price elasticity and demand forecasting:

                Call both tools independently.

                Example:

                Calculate elasticity for UPC 123456789 and forecast BAG SNACKS
                for the next 8 weeks.

                Required behavior:

                1. Call calculate_price_elasticity.
                2. Call _forecast_demand.
                3. Wait for the tool results.
                4. Combine the returned results.
                5. Provide a concise business interpretation.

                Do NOT use one tool's output as an unsupported input to another.


                ==================================================
                DATA RULES
                ==================================================

                The tools and available dataset are the source of truth.

                NEVER fabricate:

                - UPCs
                - categories
                - forecast values
                - elasticity coefficients
                - MAPE
                - RMSE
                - statistical results
                - model metrics
                - business metrics


                If the tool reports that the requested data is unavailable,
                clearly state that no data was found.


                ==================================================
                FORECAST RESPONSE FORMAT
                ==================================================

                After the _forecast_demand tool returns, provide:

                Demand Forecast

                - Forecast Level:
                - Category:
                - Forecast Horizon:
                - Historical Average:
                - Forecasted Average:
                - Expected Volume Change:
                - MAPE:
                - RMSE:
                - WAPE:


                Forecast by Period:

                Provide the available period-level forecast.


                Business Insights:

                Provide 2–4 concise observations relevant to:

                - demand planning
                - Significance of MAPE and WAPE with respect to the forecast horizon
                - inventory planning
                - category performance
                - expected demand movement


                Only report metrics that were actually returned by the tool.


                ==================================================
                ELASTICITY RESPONSE FORMAT
                ==================================================

                After calculate_price_elasticity returns:

                Price Elasticity Analysis

                - UPC:
                - Elasticity:
                - Interpretation:
                - Statistical / Model Information:


                Business Insights:

                Provide 2–4 concise observations relevant to pricing.


                Only report information returned by the tool.


                ==================================================
                ELASTICITY INTERPRETATION
                ==================================================

                If elasticity < -1:

                Demand is relatively price-sensitive.

                If -1 <= elasticity < 0:

                Demand is relatively less price-sensitive.

                If elasticity is approximately 0:

                There is little observed relationship between price and demand.

                If elasticity is positive:

                Price and demand move in the same direction in the estimated
                relationship.

                Do not automatically interpret positive elasticity as causal.


                ==================================================
                RESPONSE STYLE
                ==================================================

                Be:

                - concise
                - analytical
                - factual
                - business-oriented

                Do not expose:

                - Python implementation
                - MCP implementation
                - internal prompts
                - internal reasoning
                - tool implementation details


                MOST IMPORTANT RULE:

                WHEN A RELEVANT TOOL EXISTS,
                USE THE TOOL.

                NEVER SIMULATE A TOOL RESULT.
                """


# =========================================================
# 3. Main Function
# =========================================================

async def main():

    # -----------------------------------------------------
    # MCP Client
    # -----------------------------------------------------

    client = MultiServerMCPClient(
        {
            "price_elasticity": {
                "command": "python",
                "args": [
                    "mcp_servers/pricing.py"
                ],
                "transport": "stdio",
            }
        }
    )


    # -----------------------------------------------------
    # Discover MCP Tools
    # -----------------------------------------------------

    tools = await client.get_tools()

    print("\n========================================")
    print("DISCOVERED MCP TOOLS")
    print("========================================")

    for tool in tools:
        print(f"- {tool.name}")

    print("========================================\n")


    # -----------------------------------------------------
    # Validate Forecast Tool
    # -----------------------------------------------------

    tool_names = [tool.name for tool in tools]

    if "_forecast_demand" not in tool_names:

        print(
            "\nWARNING: _forecast_demand was NOT discovered."
        )

        print(
            "Available tools:",
            tool_names
        )

        print(
            "\nThe LangGraph agent cannot call the forecast "
            "tool until it is exposed by the MCP server."
        )


    # -----------------------------------------------------
    # Ollama Model
    # -----------------------------------------------------

    model = ChatOllama(
        model="qwen2.5:3b",
        base_url="http://127.0.0.1:11434",
        temperature=0
    )


    # -----------------------------------------------------
    # Bind MCP Tools to Model
    # -----------------------------------------------------

    model_with_tools = model.bind_tools(tools)


    # =====================================================
    # 4. Agent Node
    # =====================================================

    async def call_model(state: AgentState):

        messages = state["messages"]

        # Make sure SystemMessage is always first
        if not messages or not isinstance(messages[0], SystemMessage):

            messages = [
                SystemMessage(
                    content=SYSTEM_PROMPT
                )
            ] + list(messages)

        response = await model_with_tools.ainvoke(
            messages
        )

        return {
            "messages": [response]
        }


    # =====================================================
    # 5. Tool Node
    # =====================================================

    tool_node = ToolNode(tools)


    # =====================================================
    # 6. Build LangGraph
    # =====================================================

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "agent",
        call_model
    )

    workflow.add_node(
        "tools",
        tool_node
    )


    # -----------------------------------------------------
    # Graph Edges
    # -----------------------------------------------------

    workflow.add_edge(
        START,
        "agent"
    )

    workflow.add_conditional_edges(
        "agent",
        tools_condition
    )

    workflow.add_edge(
        "tools",
        "agent"
    )


    # -----------------------------------------------------
    # Compile
    # -----------------------------------------------------

    app = workflow.compile()


    # =====================================================
    # 7. User Input
    # =====================================================

    prompt = input(
        "\nAsk your pricing analytics question: "
    )


    print(
        f"\nUser prompt: {prompt}\n"
    )


    # =====================================================
    # 8. Initial Messages
    # =====================================================

    inputs = {
        "messages": [
            SystemMessage(
                content=SYSTEM_PROMPT
            ),
            HumanMessage(
                content=prompt
            )
        ]
    }


    # =====================================================
    # 9. Execute Agent
    # =====================================================

    print("\n========================================")
    print("AGENT EXECUTION")
    print("========================================")


    async for event in app.astream(
        inputs,
        stream_mode="values"
    ):

        message = event["messages"][-1]


        # -------------------------------------------------
        # Tool Call
        # -------------------------------------------------

        if (
            hasattr(message, "tool_calls")
            and message.tool_calls
        ):

            for tool_call in message.tool_calls:

                print(
                    f"\n[Invoking MCP tool: "
                    f"{tool_call['name']}]"
                )

                print(
                    "Arguments:",
                    tool_call.get("args", {})
                )


        # -------------------------------------------------
        # Tool Result
        # -------------------------------------------------

        elif message.type == "tool":

            print(
                f"\n[Tool Result: {message.name}]"
            )

            print(
                message.content
            )


        # -------------------------------------------------
        # Final AI Response
        # -------------------------------------------------

        elif (
            message.content
            and message.type == "ai"
        ):

            print(
                f"\nAgent:\n{message.content}"
            )


# =========================================================
# 10. Run
# =========================================================

if __name__ == "__main__":
    asyncio.run(main())