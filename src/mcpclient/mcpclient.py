from typing import Optional
from contextlib import AsyncExitStack

import boto3
from mcp import ClientSession
from mcp.client.sse import sse_client
from src.mcpclient.message import Message
from typing import List, Dict


class MCPClient:
    MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

    def __init__(self):
        # Initialize session and client objects
        self.session: List[Optional[ClientSession]] = []
        self.exit_stack = AsyncExitStack()
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        self._streams_context = []
        self.tool_session_map = {}

    async def connect_to_sse_server(self, server_url: List[str]):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        for url in server_url:
            self._streams_context.append(sse_client(url=url))

        streams = [await sc.__aenter__() for sc in self._streams_context]

        self._session_context = [ClientSession(*stream) for stream in streams]
        self.session: List[ClientSession] = [await sc.__aenter__() for sc in self._session_context]

        # Initialize
        [await s.initialize() for s in self.session]

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")

        all_tools = []
        for s in self.session:
            response = await s.list_tools()
            for r in response.tools:
                self.tool_session_map[r.name] = s
                all_tools.append(r.name)
        print("\nConnected to server with tools:", all_tools)

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    def make_bedrock_request(self, messages: List[Dict], tools: List[Dict] | None) -> Dict:
        settings = {
            "modelId": self.MODEL_ID,
            "messages": messages,
            "inferenceConfig": {"maxTokens": 1000, "temperature": 0},
        }
        if tools:
            settings["toolConfig"] = {"tools": tools}

        return self.bedrock.converse(**settings)


    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [Message.user(query).__dict__]

        response = [await s.list_tools() for s in self.session]
        available_tools = []
        for r in response:
            available_tools += [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in r.tools]
        bedrock_tools = Message.to_bedrock_format(available_tools)
        response = self.make_bedrock_request(messages, bedrock_tools)
        return await self._process_response(
          response, messages, bedrock_tools
        )

    async def _process_response(self, response: Dict, messages: List[Dict], bedrock_tools: List[Dict]) -> str:
        final_text = []
        MAX_TURNS=10
        turn_count = 0

        while True:
            if response['stopReason'] == 'tool_use':
                final_text.append("received toolUse request")
                for item in response['output']['message']['content']:
                    if 'text' in item:
                        final_text.append(f"[Thinking: {item['text']}]")
                        messages.append(Message.assistant(item['text']).__dict__)
                    elif 'toolUse' in item:
                        tool_info = item['toolUse']
                        result = await self._handle_tool_call(tool_info, messages)
                        final_text.extend(result)

                        response = self.make_bedrock_request(messages, bedrock_tools)
            elif response['stopReason'] == 'max_tokens':
                final_text.append("[Max tokens reached, ending conversation.]")
                break
            elif response['stopReason'] == 'stop_sequence':
                final_text.append("[Stop sequence reached, ending conversation.]")
                break
            elif response['stopReason'] == 'content_filtered':
                final_text.append("[Content filtered, ending conversation.]")
                break
            elif response['stopReason'] == 'end_turn':
                final_text.append(response['output']['message']['content'][0]['text'])
                break

            turn_count += 1

            if turn_count >= MAX_TURNS:
                final_text.append("\n[Max turns reached, ending conversation.]")
                break
        return "\n\n".join(final_text)

    async def _handle_tool_call(self, tool_info: Dict, messages: List[Dict]) -> List[str]:
        # (1)
        tool_name = tool_info['name']
        tool_args = tool_info['input']
        tool_use_id = tool_info['toolUseId']
        print("Calling", tool_name)

        # (2)
        result = await self.tool_session_map[tool_name].call_tool(tool_name, tool_args)

        # (3)
        messages.append(Message.tool_request(tool_use_id, tool_name, tool_args).__dict__)
        messages.append(Message.tool_result(tool_use_id, result.content).__dict__)

        # (4)
        return [f"[Calling tool {tool_name} with args {tool_args}]"]

    async def chat_loop(self):
        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def get_mcp_servers_list(self):
        response = [await s.list_tools() for s in self.session]
        available_tools = []
        for r in response:
            available_tools += [{
                "name": tool.name,
                "description": tool.description,
            } for tool in r.tools]
        return available_tools


# If you want to run seperately, please uncomment code below
# async def main():
#     if len(sys.argv) < 2:
#         print("Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>")
#         sys.exit(1)

#     client = MCPClient()
#     try:
#         await client.connect_to_sse_server(server_url="http://localhost:8000/sse")
#         await client.chat_loop()
#     finally:
#         await client.cleanup()
