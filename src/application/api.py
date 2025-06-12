import boto3
import json
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from coordinator import stream_agent_response
from fastapi import HTTPException


app = FastAPI(title="Financial Analyzer")

class PromptRequest(BaseModel):
    prompt: str
    user_id: str
    model: str

@app.post("/chat")
async def analyze(request: PromptRequest):
    prompt = request.prompt

    try:
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        return StreamingResponse(stream_agent_response(prompt, request.user_id), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def chat_history(user_id: str):
    client = boto3.client("dynamodb")
    # Query chat sessions table to get latest 20 records for user
    session_messages = client.query(
        TableName='chat_sessions',
        KeyConditionExpression='user_id = :uid',
        ExpressionAttributeValues={
            ':uid': {'S': user_id}
        },
        ScanIndexForward=False, # Sort in descending order (newest first)
        Limit=20
    )

    # Extract and format messages
    messages = [
        json.loads(item['message']['S'])
        for item in session_messages.get('Items', [])
    ]

    return JSONResponse({"messages": messages})


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "Healthy"}