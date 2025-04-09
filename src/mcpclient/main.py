import os
import sys
import json
import time
import argparse
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Literal, AsyncGenerator, Union
import uuid
import threading
from contextlib import asynccontextmanager
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.mcpclient.model import Prompt
from src.mcpclient.mcpclient import MCPClient


mcpclient = MCPClient()
async def initialize_mcp_client(app: FastAPI):
    await mcpclient.connect_to_server([
        "/Users/jinmp/Documents/allsource/mcp-bedrock/src/mcpservers/collectdata/collectdata.py",
        "/Users/jinmp/Documents/allsource/mcp-bedrock/src/mcpservers/s3/s3.py"
    ])
    yield

app = FastAPI(lifespan=initialize_mcp_client)



# Health Check
@app.get("/")
def health_check():
    return JSONResponse({"status": "healthy"})


# Chat API
@app.post("/chat")
async def chat_stream(request: Request, prompt: Prompt):
    try:
        prompt = prompt.content

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        async def generate_stream():
            messages = [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
            try:
                response = mcpclient.make_bedrock_request(messages=messages, tools=None)

                # Handle the EventStream response
                for event in response['stream']:
                    if 'contentBlockDelta' in event:
                        chunk = event['contentBlockDelta']['delta'].get('text', '')
                        if chunk:
                            # Format for SSE
                            yield f"{json.dumps({'content': chunk})}\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_mcp_servers")
async def get_mcp_servers(request: Request):
    try:
        # Get the list of MCP servers
        mcp_servers = mcpclient.get_mcp_servers_list()
        # Return the list of MCP servers
        return JSONResponse(mcp_servers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_mcp_servers")
async def set_mcp_servers(request: Request, server_setting: dict):
    try:
        mcpclient.set_mcp_servers(server_setting)
        return JSONResponse({"status": "success"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test_mcpclient")
async def test_mcpclient(request: Request, prompt: Prompt):
    try:
        prompt = prompt.content
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # Use the mcpclient to process the prompt
        response = await mcpclient.process_query(prompt)

        # Return the response
        return JSONResponse({"response": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))