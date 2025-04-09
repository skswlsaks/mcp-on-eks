from datetime import datetime, timedelta
from enum import Enum
import json
from typing import Sequence

from zoneinfo import ZoneInfo
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("time")

from pydantic import BaseModel



@mcp.tool()
async def get_current_time(timezone: str) -> str:
    """Get current time in specified timezone

    Args:
        timezone:

    """
    time = datetime.now()
    return "\n---\n".join(forecasts)
