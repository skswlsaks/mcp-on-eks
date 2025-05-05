import logging
import sys
import mcp_log as log
import mcp_rag as rag
import mcp_s3 as storage
import mcp_coder as coder

from mcp.server.fastmcp import FastMCP 

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-server")

try:
    mcp = FastMCP(
        name = "tools",
        instructions=(
            "You are a helpful assistant. "
            "You can use tools for the user's question and provide the answer."
        ),
    )
    logger.info("MCP server initialized successfully")
except Exception as e:
        err_msg = f"Error: {str(e)}"
        logger.info(f"{err_msg}")

######################################
# RAG
######################################
@mcp.tool()
def search(keyword: str) -> str:
    """
    Search the knowledge base with the given keyword.
    keyword: the keyword to search
    return: the result of search
    """
    logger.info(f"search --> keyword: {keyword}")

    return rag.retrieve_knowledge_base(keyword)

######################################
# Code Interpreter
######################################
@mcp.tool()
def repl_coder(code):
    """
    Use this to execute python code and do math. 
    If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.
    code: the Python code was written in English
    """
    logger.info(f"repl_coder --> code: {code}")
    
    return coder.repl_coder(code)

@mcp.tool()
def repl_drawer(code):
    """
    Execute a Python script for draw a graph.
    Since Python runtime cannot use external APIs, necessary data must be included in the code.
    The graph should use English exclusively for all textual elements.
    Do not save pictures locally bacause the runtime does not have filesystem.
    When a comparison is made, all arrays must be of the same length.
    code: the Python code was written in English
    return: the url of graph
    """ 
    logger.info(f"repl_drawer --> code: {code}")
        
    return coder.repl_drawer(code)

######################################
# AWS Logs
######################################
@mcp.tool()
async def list_groups(
    prefix: Optional[str] = None,
    region: Optional[str] = 'us-west-2'
) -> str:
    """List available CloudWatch log groups."""
    logger.info(f"list_groups --> prefix: {prefix}, region: {region}")

    return await log.list_groups(prefix=prefix, region=region)

@mcp.tool()
async def get_logs(
    logGroupName: str,
    logStreamName: Optional[str] = None,
    startTime: Optional[str] = None,
    endTime: Optional[str] = None,
    filterPattern: Optional[str] = None,
    region: Optional[str] = 'us-west-2'
) -> str:
    """Get CloudWatch logs from a specific log group and stream."""
    logger.info(f"get_logs --> logGroupName: {logGroupName}, logStreamName: {logStreamName}, startTime: {startTime}, endTime: {endTime}, filterPattern: {filterPattern}, region: {region}")

    return await log.get_logs(
        logGroupName=logGroupName,
        logStreamName=logStreamName,
        startTime=startTime,
        endTime=endTime,
        filterPattern=filterPattern,
        region=region
    )

######################################
# AWS S3
######################################
from typing import List, Optional
from mcp.types import Resource

@mcp.tool()
async def list_buckets(
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List S3 buckets using async client with pagination
    """
    logger.info(f"list_buckets --> start_after: {start_after}, max_buckets: {max_buckets}, region: {region}")

    return await storage.list_buckets(start_after, max_buckets, region)
    
@mcp.tool()  
async def list_objects(
    bucket_name: str, 
    prefix: Optional[str] = "", 
    max_keys: Optional[int] = 1000,
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List objects in a specific bucket using async client with pagination
    Args:
        bucket_name: Name of the S3 bucket
        prefix: Object prefix for filtering
        max_keys: Maximum number of keys to return,
        region: Name of the aws region
    """
    logger.info(f"list_objects --> bucket_name: {bucket_name}, prefix: {prefix}, max_keys: {max_keys}, region: {region}")

    return await storage.list_objects(bucket_name, prefix, max_keys, region)

@mcp.tool()    
async def list_resources(
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> List[Resource]:
    """
    List S3 buckets and their contents as resources with pagination
    Args:
        start_after: Start listing after this bucket name
    """
    logger.info(f"list_resources --> start_after: {start_after}, max_buckets: {max_buckets}, region: {region}")
    
    return await storage.list_resources(start_after, max_buckets, region)
    
######################################
# AWS Logs
######################################

if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")


