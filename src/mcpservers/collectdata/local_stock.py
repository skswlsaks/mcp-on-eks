import os
import requests
import yfinance as yf
import httpx
from datetime import date, timedelta, datetime
from fastmcp import FastMCP
import json


POLYGON_APIKEY = os.environ.get("POLYGON_APIKEY") if os.environ.get("POLYGON_APIKEY") else ""

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
async def get_stock_price_data(ticker_symbol, start_date, end_date=date.today()):
    """
    Fetch historical price data for a given stock ticker using Yahoo Finance.
    Time difference between start_date and end_date should be less than 180 days.

    Args: \n
        `ticker_symbol` (str): Stock ticker symbol
        `start_date` (str): Startdate in format YYYY-MM-DD
        `end_date` (str): Enddate in format YYYY-MM-DD, by default this is set to today

    Returns: \n
        Close(c), High(h), Low(l), Open(o) and Volume Weighted(vw) price of stock. Number of transactions(n), Trading Volume(v) are also provided.
    """
    end_date = date.fromisoformat(end_date)
    start_date = date.fromisoformat(start_date)

    try:
        response = requests.get(f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=180&apiKey={POLYGON_APIKEY}")
        results = json.loads(response.text)['results']
        print(results)
        # # Create ticker object
        # ticker = yf.Ticker(ticker_symbol)

        # # Get historical data
        # df = ticker.history(start=start_date, end=end_date)
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None
    return results
    # Reset index to make Date a column
    # df = df.reset_index()
    # return df.to_json()


@mcp.tool()
async def get_current_time():
    """
    Fetches the current time of machine

    Returns: \n
        str: Current time in ISO format
    """
    return datetime.now().isoformat()


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
    return float(current_yield)


if __name__ == "__main__":
    mcp.run()