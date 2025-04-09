import datetime
import yfinance as yf
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("stock")


@mcp.tool()
async def crawl_companyfacts(cik: str) -> str:
    """
    Fetches company facts data from SEC EDGAR API for a given CIK number.

    Args:
        cik (str): The Central Index Key (CIK) number for the company, 10 digits

    Returns:
        str: JSON response containing company facts data if successful,
             error message string if request fails
    """
    async with httpx.AsyncClient() as client:
        # First get the company filing index
        fact_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        response = await client.get(fact_url, headers={"User-Agent": "Amazon jinmp@amazon.com"})
        if response.status_code != 200:
            return f"Failed to fetch index page: {response.status_code}"

        return response.content

@mcp.tool()
async def map_ticker_to_cik(ticker: str) -> str:
    """
    Maps a stock ticker symbol to its SEC CIK number.

    Args:
        ticker (str): Stock ticker symbol (e.g. 'AAPL' for Apple)

    Returns:
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
            print(entry)
            if entry['ticker'] == ticker:
                # Format CIK with leading zeros to 10 digits
                return str(entry['cik_str']).zfill(10)

        return f"Could not find CIK for ticker {ticker}"

@mcp.tool()
async def get_stock_price_data(ticker_symbol, start_date=None, end_date=datetime.datetime.today()):
    """
    Fetch historical price data for a given stock ticker using Yahoo Finance

    Args:
        ticker_symbol (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format (default: 60 days ago)
        end_date (str): End date in YYYY-MM-DD format, today's date will become an input if end_date is not specified (default: today)

    Returns:
        pandas.DataFrame: DataFrame containing price data
    """
    # Set default dates if not provided
    if not start_date:
        start_date = end_date - datetime.timedelta(days=60)

    # Create ticker object
    ticker = yf.Ticker(ticker_symbol)

    # Get historical data
    df = ticker.history(start=start_date, end=end_date)

    # Reset index to make Date a column
    df = df.reset_index()
    return df




# if __name__ == "__main__":
#     import asyncio
    # Run the async function
    # result = asyncio.run(crawl_companyfacts("0000320193"))
    # result = asyncio.run(map_ticker_to_cik("AAPL"))
    # print(result)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
