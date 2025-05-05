import sys
import httpx
import logging
import pandas as pd
from datetime import date, timedelta
import yfinance as yf

from fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("aws-cli")

try:
    mcp = FastMCP(
        name = "stock_data",
        instructions=(
            "You are a helpful assistant. "
            "You can use tools for the user's question and provide the answer."
        ),
    )
    logger.info("MCP server initialized successfully")
except Exception as e:
        err_msg = f"Error: {str(e)}"
        logger.info(f"{err_msg}")



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
    return df.to_json()



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


if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")

