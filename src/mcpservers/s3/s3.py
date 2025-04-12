import click
import boto3
import uvicorn
from datetime import datetime
from io import BytesIO
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route

mcp = FastMCP("s3")

@mcp.tool()
async def save_text_to_s3(text: str, bucket_name: str, file_name: str) -> str:
    """
    Save text content to a txt file and upload to S3

    Args: \n
        `text` (str): Text content to save
        `bucket_name` (str): Name of S3 bucket
        `file_name` (str): Name of file to create in S3

    Returns:
        str: Response containing status and file details
    """
    s3_client = boto3.client('s3')

    try:
        # Write within memory
        memoryfile = BytesIO()
        memoryfile.write(text.encode('utf-8'))
        memoryfile.seek(0)

        # Upload file to S3
        s3_client.upload_fileobj(
            memoryfile,
            bucket_name,
            file_name,
            ExtraArgs={
                'ContentType': 'text/plain',
                'Metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'content-length': str(len(text))
                }
            }
        )

        # Get the file URL
        file_url = f"s3://{bucket_name}/{file_name}"

        return {
            "status": "success",
            "message": "File saved successfully",
            "file_details": {
                "bucket": bucket_name,
                "filename": file_name,
                "size": len(text),
                "location": file_url
            }
        }

    except Exception as e:
        print(f"Error saving file to S3: {str(e)}")
        raise e
    finally:
        memoryfile.close()


@click.command()
@click.option("--port", default=8080)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main_server(port: int, transport: str) -> int:

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0], streams[1], mcp._mcp_server.create_initialization_options()
            )

    starlette_app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    uvicorn.run(starlette_app, host="0.0.0.0", port=port)

    return 0

if __name__ == "__main__":
    main_server()
