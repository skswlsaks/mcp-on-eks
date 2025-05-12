from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse


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

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy"})

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, path="/messages")

