import click
import uvicorn
from datetime import date, timezone, datetime, timedelta
import yfinance as yf
import httpx
from fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route


# Initialize FastMCP server
mcp = FastMCP("stock_data")


@mcp.tool()
async def crawl_companyconcept(cik: str, concept: str) -> str:
    """
    Fetches company concept data from SEC EDGAR API for a given CIK number.

    Args: \n
        `cik` (str): The Central Index Key (CIK) number for the company, 10 digits
        `concept` (str): The company concept to fetch data for (e.g. Assets, Liabilities, etc.)

    Returns: \n
        str: JSON response containing the most recent 5 company concept data if successful,
             error message string if request fails
    """
    async with httpx.AsyncClient() as client:
        # First get the company filing index
        fact_url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
        response = await client.get(fact_url, headers={"User-Agent": "Amazon jinmp@amazon.com"})
        if response.status_code != 200:
            return f"Failed to fetch index page: {response.status_code}"

        vals = response.json()['units']
        # Get last 5 records
        units = list(vals.keys())[0]
        return vals[units][-1]

@mcp.tool()
async def map_ticker_to_cik(ticker: str) -> str:
    """
    Maps a stock ticker symbol to its SEC CIK number.

    Args: \n
        `ticker` (str): Stock ticker symbol (e.g. 'AAPL' for Apple)

    Returns: \n
        str: 10-digit CIK number if found,
             error message string if ticker cannot be mapped
    """

    # Create async HTTP client
    async with httpx.AsyncClient() as client:
        # Call SEC API to get CIK from ticker
        url = "https://www.sec.gov/files/company_tickers.json"
        response = await client.get(url, headers={"User-Agent": "Amazon jinmp@amazon.com"})

        if response.status_code != 200:
            return f"Failed to fetch ticker data: {response.status_code}"

        # Parse response
        ticker_data = response.json()

        # Search for ticker (case insensitive)
        ticker = ticker.upper()
        for entry in ticker_data.values():
            if entry['ticker'] == ticker:
                # Format CIK with leading zeros to 10 digits
                return str(entry['cik_str']).zfill(10)

        return f"Could not find CIK for ticker {ticker}"

@mcp.tool()
async def get_stock_price_data(ticker_symbol):
    """
    Fetch historical price data for a given stock ticker using Yahoo Finance

    Args: \n
        `ticker_symbol` (str): Stock ticker symbol

    Returns: \n
        pandas.DataFrame: DataFrame containing price data
    """
    end_date = date.today()
    # Set default dates if not provided
    start_date = end_date - timedelta(days=60)

    # Create ticker object
    ticker = yf.Ticker(ticker_symbol)

    # Get historical data
    df = ticker.history(start=start_date, end=end_date)

    # Reset index to make Date a column
    df = df.reset_index()
    return df


@mcp.tool()
async def get_moodys_aaa_bond_yield() -> str:
    """
    Fetches the current Moody's AAA bond yield from Yahoo Finance

    Returns: \n
        str: Current Moody's AAA bond yield as a string
    """
    # Create ticker object for Moody's AAA bond yield (^IRX)
    ticker = yf.Ticker("^IRX")

    # Get the latest data point
    data = ticker.history(period="1d")

    # Extract the current bond yield
    current_yield = data['Close'].iloc[-1]
    f"Current Moody's AAA bond yield: {current_yield:.2f}"
    return float(current_yield)


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
