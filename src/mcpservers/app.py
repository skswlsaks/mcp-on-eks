import contextlib
from fastapi import FastAPI
from collectdata.collectdata import mcp as collectdata_mcp

# # Create a combined lifespan to manage both session managers
# @contextlib.asynccontextmanager
# async def lifespan(app: FastAPI):
#     async with contextlib.AsyncExitStack() as stack:
#         await stack.enter_async_context(collectdata_mcp.run(transport='streamable-http'))
#         yield

app = FastAPI()
app.mount("/collectdata", collectdata_mcp.http_app("/collectdata"))
# app.mount("/math", math.mcp.streamable_http_app())