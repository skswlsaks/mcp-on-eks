import os
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent, tool
from strands_tools import retrieve
from bedrock_models import sonnetv2
from strands.tools.mcp import MCPClient


DATA_COLLECTOR_SYSTEM_PROMPT = """
You are a specialized Stock Data Collection Agent designed to support stock analysis systems.
You need to justify the time using get_current_time. And get data based on current time.

Your core responsibilities:
1. Act as the primary data gathering interface
2. Collect and validate financial market data including:
   - Stock prices and trading volumes
   - Company financial metrics
   - Market indicators
   - News and events affecting stocks
   - Industry and sector data

You must:
- Maintain data accuracy and reliability
- Provide clear source attribution
- Format data in a structured, analysis-ready format
- Flag any data inconsistencies or gaps
- Ensure data is properly timestamped and versioned
- Support real-time and historical data requests

When responding:
- Be precise and factual
- Include relevant metadata
- Indicate data freshness and reliability
- Structure responses for easy integration with analysis systems
- Alert if requested data is unavailable or incomplete

You operate as part of a larger stock analysis system, focusing solely on data collection and preparation. Your output will be used by analysis agents for further processing and insights generation.
"""

DATA_MCP_URL = os.environ.get("DATA_MCP_URL") if os.environ.get("DATA_MCP_URL") else "http://localhost:8000/messages/"

mcp_http_client = MCPClient(lambda: streamablehttp_client(DATA_MCP_URL))

@tool
def security_data_collector(query: str) -> str:
    """
    This tool is used to collect needed financial data.
    """

    try:
        with mcp_http_client:
            tools = mcp_http_client.list_tools_sync() + [retrieve]

            data_collector = Agent(
                model=sonnetv2,
                system_prompt=DATA_COLLECTOR_SYSTEM_PROMPT,
                tools=tools,
                max_parallel_tools=2
            )
            return data_collector(query)
    except Exception as e:
        return f"Error in data collector: {e}"
