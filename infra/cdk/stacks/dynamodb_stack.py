from aws_cdk import (
    Stack,
    aws_dynamodb as dynamodb,
    RemovalPolicy,
)
from constructs import Construct

class DynamoDBStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create a DynamoDB table with auto-scaling
        table = dynamodb.Table(self, "McpDynamoDBTable",
            partition_key=dynamodb.Attribute(
                name="user_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.NUMBER
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,  # On-demand capacity (auto-scaling)
            removal_policy=RemovalPolicy.RETAIN,  # Protect against accidental deletion
            point_in_time_recovery=True,  # Enable point-in-time recovery
            table_name="chat_sessions"
        )

        # Export the table name
        self.table = table