
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from src.mcpclient.model import Prompt
from src.mcpclient.mcpclient import MCPClient

mcp_servers = {
    "s3": "http://localhost:8080/sse",
    "financial_data": "http://localhost:8081/sse",
    "financial_analysis": "http://localhost:8082/sse",
}

mcpclient = MCPClient()
async def initialize_mcp_client(app: FastAPI):
    await mcpclient.connect_to_sse_server(list(mcp_servers.values()))
    yield

app = FastAPI(lifespan=initialize_mcp_client)


# Health Check
@app.get("/")
def health_check():
    return JSONResponse({"status": "healthy"})


# Chat API
@app.post("/chat")
async def chat(request: Request, prompt: Prompt):
    try:
        prompt = prompt.content
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # Use the mcpclient to process the prompt
        response = await mcpclient.process_query(prompt)

        # Return the response
        return JSONResponse({"response": response})

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_mcp_servers")
async def get_mcp_servers(request: Request):
    try:
        # Get the list of MCP servers
        mcp_servers = await mcpclient.get_mcp_servers_list()
        # Return the list of MCP servers
        return JSONResponse(mcp_servers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

