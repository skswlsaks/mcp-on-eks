from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
    aws_bedrock as bedrock,
    CfnOutput,
    RemovalPolicy,
)
from constructs import Construct

class BedrockStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create S3 bucket for knowledge base data
        kb_bucket = s3.Bucket(self, "BedrockKnowledgeBaseBucket",
            bucket_name=f"mcp-bedrock-kb-{self.account}-{self.region}",
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True
        )

        # Create an IAM role that can be assumed by the EKS cluster to access Bedrock
        bedrock_role = iam.Role(self, "BedrockAccessRole",
            assumed_by=iam.ServicePrincipal("eks.amazonaws.com")
        )
        
        # Create a policy for Bedrock access with Claude model
        bedrock_policy = iam.Policy(self, "BedrockPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "bedrock:InvokeModel",
                        "bedrock:InvokeModelWithResponseStream"
                    ],
                    resources=[
                        # Claude models ARNs
                        f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-v2",
                        f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-v2:1",
                        f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
                        f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
                    ]
                ),
                iam.PolicyStatement(
                    actions=[
                        "bedrock:ListFoundationModels",
                        "bedrock:GetFoundationModel",
                        "bedrock:CreateKnowledgeBase",
                        "bedrock:GetKnowledgeBase",
                        "bedrock:UpdateKnowledgeBase",
                        "bedrock:DeleteKnowledgeBase",
                        "bedrock:ListKnowledgeBases",
                        "bedrock:CreateDataSource",
                        "bedrock:GetDataSource",
                        "bedrock:UpdateDataSource",
                        "bedrock:DeleteDataSource",
                        "bedrock:ListDataSources",
                        "bedrock:StartIngestionJob",
                        "bedrock:GetIngestionJob",
                        "bedrock:ListIngestionJobs",
                        "bedrock:Retrieve"
                    ],
                    resources=["*"]
                ),
                iam.PolicyStatement(
                    actions=[
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:ListBucket",
                        "s3:DeleteObject"
                    ],
                    resources=[
                        kb_bucket.bucket_arn,
                        f"{kb_bucket.bucket_arn}/*"
                    ]
                )
            ]
        )
        
        # Attach the policy to the role
        bedrock_policy.attach_to_role(bedrock_role)
        
        # Create a Bedrock Knowledge Base
        knowledge_base = bedrock.CfnKnowledgeBase(self, "McpKnowledgeBase",
            name="mcp-knowledge-base",
            role_arn=bedrock_role.role_arn,
            knowledge_base_configuration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v1"
                }
            },
            storage_configuration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": kb_bucket.bucket_arn
                }
            }
        )
        
        # Output the role ARN and bucket name
        CfnOutput(self, "BedrockRoleArn",
            value=bedrock_role.role_arn,
            description="ARN of the IAM role for Bedrock access"
        )
        
        CfnOutput(self, "KnowledgeBaseBucket",
            value=kb_bucket.bucket_name,
            description="S3 bucket for Bedrock Knowledge Base data"
        )
        
        CfnOutput(self, "KnowledgeBaseId",
            value=knowledge_base.attr_knowledge_base_id,
            description="ID of the Bedrock Knowledge Base"
        )