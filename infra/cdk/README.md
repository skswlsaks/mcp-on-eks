# MCP Workshop CDK Infrastructure

This CDK project deploys the following AWS resources:

- EKS cluster with auto-scaling node groups
- DynamoDB table with on-demand capacity (auto-scaling)
- Bedrock with Claude model access

## Prerequisites

- AWS CLI configured with appropriate credentials
- Node.js and npm installed
- Python 3.8+ installed
- CDK installed globally: `npm install -g aws-cdk`

## Setup

1. Create a virtual environment:
```
python -m venv .venv
```

2. Activate the virtual environment:
```
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Bootstrap CDK (if not already done):
```
cdk bootstrap
```

## Deployment

To deploy all stacks:
```
cdk deploy --all
```

To deploy individual stacks:
```
cdk deploy McpEksStack
cdk deploy McpDynamoDBStack
cdk deploy McpBedrockStack
```

## Useful CDK Commands

* `cdk ls`          list all stacks in the app
* `cdk synth`       emits the synthesized CloudFormation template
* `cdk deploy`      deploy this stack to your default AWS account/region
* `cdk diff`        compare deployed stack with current state
* `cdk destroy`     destroy the deployed stack