import boto3
import json
import time
from strands import Agent
from strands_tools import current_time
from strands.agent.conversation_manager import SlidingWindowConversationManager
from bedrock_models import sonnetv2
from datacollector import security_data_collector
from analysis import financial_analysis

COORDINATOR_SYSTEM_PROMPT = """
You are a executive coordinator who analyse or return the financial task.
You have to split the task into small steps and each steps need to be solved by appropriate agents.
You are given a list of agents and their actions.
You must decide which agent should perform the action.
For gathering all types of financial, you can use data collector agent.
For doing financial analysis, you can use analysis agent based on data gathered from data collector agent.
You must use get_current_time to fetch current and then do certain data related work.

Your process should be:
1. Determine which type of task is user asking to solve
2. If the task is simply collecting required data, it should be passed to data collector agent.
3. If the task needs insights and deep financial analysis then it should be passed to analysis agent.
4. If the analysis agent needs back up data, it should collect needed data from data collector agent.
5. If the task is non-financial related, then simply answer the question using common sense by the coordinator.

The result should in markdown understandable format.
"""

client = boto3.client('dynamodb')
conversation_manager = SlidingWindowConversationManager(window_size=10)

async def stream_agent_response(prompt, user_id):

    # Query chat sessions table to get latest 20 records for user
    message_history = client.query(
        TableName='chat_sessions',
        KeyConditionExpression='user_id = :uid',
        ExpressionAttributeValues={
            ':uid': {'S': user_id}
        },
        ScanIndexForward=False, # Sort in descending order (newest first)
        Limit=10
    )

    client.put_item(
        TableName='chat_sessions',
        Item={
            'user_id': {'S': user_id},
            'timestamp': {'N': str(int(time.time()))},
            'message': {'S': json.dumps({'role': 'user', 'content': [{'text': prompt}]})}
        }
    )

    coordinator = Agent(
        model=sonnetv2,
        system_prompt=COORDINATOR_SYSTEM_PROMPT,
        # - security_data_collector: retrieve needed financial data from source
        # - financial_analysis: analyze data to get insights or produce report
        # - current_time: to get system current time
        # - retrieve: to get latest knowledge from Bedrock Knowledge Base
        tools=[security_data_collector, financial_analysis, current_time],
        conversation_manager=conversation_manager,
        callback_handler=None,
        messages=[json.loads(m['message']['S']) for m in message_history['Items']],
    )

    tool_flag = False
    tool_name = None
    tool_input = None
    tool_result = None

    async for event in coordinator.stream_async(prompt):
        res = None

        # Handle regular text content
        if 'data' in event:
            res = {"type": "message", "content": event['data']}

        # Handle tool call start
        elif 'event' in event and 'contentBlockStart' in event['event']:
            if 'toolUse' in event['event']['contentBlockStart'].get('start', {}):
                tool_flag = True
                tool_name = event['event']['contentBlockStart']['start']['toolUse'].get('name', '')
                tool_id = event['event']['contentBlockStart']['start']['toolUse'].get('toolUseId', '')
                res = {"type": "tool_start", "name": tool_name, "tool_id": tool_id}

        # Handle tool input parameters
        elif 'current_tool_use' in event:
            tool_name = event['current_tool_use'].get('name', '')
            tool_input = event['current_tool_use'].get('input', '')
            res = {"type": "tool_input", "name": tool_name, "input": tool_input}

        # Handle tool result
        elif 'toolResult' in event.get('message', {}).get('content', [{}])[0]:
            tool_result = event['message']['content'][0]['toolResult'].get('content', [{}])[0].get('text', '')
            tool_id = event['message']['content'][0]['toolResult'].get('toolUseId', [{}])
            res = {"type": "tool_result", "name": tool_name, "result": tool_result, "tool_id": tool_id}

        elif 'message' in event:
            client.put_item(
                TableName='chat_sessions',
                Item={
                        'user_id': {'S': user_id},
                        'timestamp': {'N': str(int(time.time()))},
                        'message': {'S': json.dumps(event)}
                    }
            )

        if res:
            yield json.dumps(res)+'\n'

