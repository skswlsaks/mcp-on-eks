#!/usr/bin/env python3
import os
from aws_cdk import App, Environment

from stacks.eks_stack import EksStack
from stacks.dynamodb_stack import DynamoDBStack
from stacks.bedrock_stack import BedrockStack

app = App()

# Define your AWS environment
env = Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT", ""),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1")
)

# Create the stacks
eks_stack = EksStack(app, "McpEksStack", env=env)
dynamodb_stack = DynamoDBStack(app, "McpDynamoDBStack", env=env)
bedrock_stack = BedrockStack(app, "McpBedrockStack", env=env)

app.synth()