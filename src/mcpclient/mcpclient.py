import sys
import asyncio
from typing import Optional, List, Dict
from contextlib import AsyncExitStack

import boto3
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.mcpclient.message import Message


class MCPClient:
    MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        self.tools = []

    async def connect_to_server(self, server_script_path: List[str]):
        for script in server_script_path:
            if not script.endswith(('.py', '.js')):
                raise ValueError("Server script must be a .py or .js file")
            # command = "python" if script.endswith('.py') else "node"
        server_params = StdioServerParameters(command="python", args=server_script_path, env=None)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        response = await self.session.list_tools()
        self.tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema,
            "status": True
        } for tool in response.tools]
        print("\nConnected to server with tools:", [tool.name for tool in response.tools])

    async def cleanup(self):
        await self.exit_stack.aclose()

    def make_bedrock_request(self, messages: List[Dict], tools: List[Dict] | None) -> Dict:
        if tools:
            return self.bedrock.converse(
                modelId=self.MODEL_ID,
                messages=messages,
                inferenceConfig={"maxTokens": 1000, "temperature": 0},
                toolConfig={"tools": tools}
            )
        else:
            return self.bedrock.converse_stream(
                modelId=self.MODEL_ID,
                messages=messages,
                inferenceConfig={"maxTokens": 1000, "temperature": 0}
            )

    async def process_query(self, query: str) -> str:
        messages = [Message.user(query).__dict__]
        response = await self.session.list_tools()
        bedrock_tools = Message.to_bedrock_format(list(filter(lambda x: x['status'], self.tools)))
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

        # (2)
        result = await self.session.call_tool(tool_name, tool_args)

        # (3)
        messages.append(Message.tool_request(tool_use_id, tool_name, tool_args).__dict__)
        messages.append(Message.tool_result(tool_use_id, result.content).__dict__)

        # (4)
        return [f"[Calling tool {tool_name} with args {tool_args}]"]


    async def chat_loop(self):
        print("\nMCP Client Started!\nType your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    def get_mcp_servers_list(self):
        resp = {}
        for tool in self.tools:
            resp[tool["name"]] = tool["status"]
        return resp

    def set_mcp_servers(self, server_settings: dict):
        print(self.tools)
        for i in range(len(self.tools)):
            self.tools[i]["status"] = server_settings[self.tools[i]["name"]]


# client.py
# async def main():
#     if len(sys.argv) < 2:
#         print("Usage: python client.py <path_to_server_script>")
#         sys.exit(1)

#     client = MCPClient()
#     try:
#         await client.connect_to_server(sys.argv[1])
#         await client.chat_loop()
#     finally:
#         await client.cleanup()

# if __name__ == "__main__":
#     asyncio.run(main())