from typing import List, Optional
import aioboto3
from mcp.server.fastmcp import FastMCP

session = aioboto3.Session()

mcp = FastMCP("tools")

@mcp.tool()
async def list_buckets(
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List S3 buckets using async client with pagination
    """
    async with session.client('s3', region_name=region) as s3:
        # Default behavior if no buckets configured
        response = await s3.list_buckets()
        buckets = response.get('Buckets', [])

        if start_after:
            buckets = [b for b in buckets if b['Name'] > start_after]

        return buckets[:max_buckets]

if __name__ =="__main__":
    mcp.run(transport="stdio")

