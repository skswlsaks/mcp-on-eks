import click
import uvicorn
from fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route


# Initialize FastMCP server
mcp = FastMCP("stock_analysis")

@mcp.tool()
async def calculate_intrinsic_value(eps: float, yield_aaa_bond: float, growth_rate=10.5) -> float:
    """
    Calculate the intrinsic value of a stock based on earnings per share (EPS),
    growth rate and Moody's Seasoned AAA Bond Yield.

    Args: \n
        `ticker` (str): Stock ticker symbol (e.g. 'AAPL' for Apple)
        `eps` (float): Earning per share of stock ticker
        `yield_aaa_bond` (float): Moody's Seasoned AAA Bond Yield as a percentage (e.g. 2.5 for 2.5%)

    Returns:
        float: The calculated intrinsic value of the stock.
    """
    # Calculate intrinsic value
    intrinsic_value = eps * (8.5 + 2 * growth_rate) * 4.4 / yield_aaa_bond
    return intrinsic_value

@mcp.tool()
async def should_buy_or_not(ticker: str, current_price: float, intrinsic_price: float) -> str:
    """
    Determine whether to buy or not based on the current price and intrinsic price.

    Args: \n
        `current_price` (float): Current price of the stock
        `intrinsic_price` (float): Intrinsic price of the stock

    Returns:
        str: A string indicating whether to buy or not
    """
    if current_price < intrinsic_price:
        return f"Buy {ticker} stock, current price is {current_price} and intrinsic price is {intrinsic_price}"
    else:
        return f"Don't buy {ticker} stock, current price is {current_price} and intrinsic price is {intrinsic_price}"


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
            Route("/", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    uvicorn.run(starlette_app, host="0.0.0.0", port=port)

    return 0

if __name__ == "__main__":
    main_server()