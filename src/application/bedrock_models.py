import os
import json
import requests
from strands.models import BedrockModel

# data = {
#     "inferenceProfileName": "infbedrock_role",
#     "modelSource": {
#         "copyFrom": "arn:aws:bedrock:us-east-1::foundation-model/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
#     }
# }
# respons = requests.post("https://bedrock.us-east-1.amazonaws.com/inference-profiles", json=data)
# print(respons.text)

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID") if os.environ.get("BEDROCK_MODEL_ID") else "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
BEDROCK_MODEL_REGION = os.environ.get("BEDROCK_MODEL_REGION") if os.environ.get("BEDROCK_MODEL_REGION") else "us-west-2"

# Create a Bedrock model with the custom session
sonnetv2 = BedrockModel(
    model_id=BEDROCK_MODEL_ID,
    region_name="us-west-2",
    temperature=0.3,
    top_p=0.8,
)
