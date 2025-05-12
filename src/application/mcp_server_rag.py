import os
import logging
import sys
import requests

from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("rag")

# rag_url = os.environ.get("RAGURL") if os.environ.get("RAGURL") else "http://mcp-rag-service.mcp-server.svc.cluster.local"
rag_url = os.environ.get("RAGURL") if os.environ.get("RAGURL") else "http://localhost:8000"
kb_name = os.environ.get("KBNAME") if os.environ.get("KBNAME") else "kb-mcp-demo"

try:
    mcp = FastMCP(
        name = "rag",
        instructions=(
            "You are a helpful assistant. "
            "You retrieve documents in RAG."
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
def rag_search(keyword: str) -> str:
    """
    Search the knowledge base with the given keyword.
    keyword: the keyword to search
    return: the result of search
    """
    logger.info(f"search --> keyword: {keyword}")

    response = requests.post(rag_url, json={
        "knowledge_base_name": kb_name,
        "keyword": keyword,
        "top_k": 3
    })
    return response.text

if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")


