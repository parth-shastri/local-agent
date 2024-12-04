from datetime import datetime as dt
import requests
from typing import Annotated
import json
import yfinance as yf
from pydantic import Field, ValidationError
import regex as re
from duckduckgo_search import DDGS
from src.tools.functions.output_types import CompanyName, Ticker


def get_recent_news(
    ticker: Annotated[str, "company symbol"],
    company_name: Annotated[str, "company name"],
):
    """
    Get recent news for a company & ticker using internet search.
    Args:
        ticker (str): The stock ticker symbol to search for news.

    Returns:
        str: A formatted string containing recent news articles, including titles,
             article snippets, and source URLs.

    Example:
        >>> news = get_recent_news("AAPL")
        >>> print(news)
        ## Recent news for AAPL
        Title: Apple Reports Third Quarter Results
        Article: Apple today announced financial results for its fiscal 2023 third quarter...
        Sources: https://www.apple.com/newsroom/2023/08/apple-reports-third-quarter-results/
        ...
    """
    news_search = DDGS().news
    news_list = news_search(
        f"{company_name}, {ticker} performance analysis {dt.datetime.now().strftime('%B %Y')}",
        region="in",
        max_results=5,
        timelimit="w",
    )
    news = f"\n## Recent news for {ticker}\n"
    for doc in news_list:
        news += f"Title: {doc['title']}\n"
        news += f"Date: {doc['date']}\n"
        news += f"Article: {doc['body']}\n"
        news += f"Source: {doc['url']}\n"

    return news


def stock_search(company_name: str) -> str:
    """Searches for a stock ticker symbol based on a company name.

    Args:
        company_name (str): The name of the company to search for a stock ticker symbol.

    Returns:
        str: The stock ticker symbol associated with the company name.
    """
    try:
        yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        params = {"q": company_name, "quotes_count": 1, "country": "United States"}

        res = requests.get(url=yfinance, params=params, headers={"User-Agent": user_agent})
        data = res.json()

        company_code = data["quotes"][0]["symbol"]
        # The company code comes as "companyname.NS" or "companyname.BO", remove the ['NS', 'BO']
        company_code = company_code.split(".")[0]
        return company_code
    except Exception as e:
        print(f"Error searching for stock ticker: {e}")
        return "Error searching for stock ticker, please use a different company"


# yahoo finance financials fetcher
def yf_get_financial_statements(
    ticker: str = Field(description="the ticker/trading symbol of the company"),
) -> str:
    """Fetches the financial statements data from yahoo finance.

    Args:
        ticker (str): The ticker symbol of the company to fetch information for.

    Returns:
        str: A string containing the financial statement data in a formatted way.
    """

    if "." in ticker:
        ticker = ticker.split(".")[0]

    ticker += ".NS"
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet
    if balance_sheet.shape[-1] > 3:
        balance_sheet = balance_sheet.iloc[:, :3]

    balance_sheet = balance_sheet.dropna(how="any")
    # Convert the sheet to Cr.
    balance_sheet = balance_sheet.multiply(1e-7)
    # rename the index
    balance_sheet.rename(
        index={name: f"{name} (in Crores.)" for name in balance_sheet.index},
        inplace=True,
    )
    # convert to string
    balance_sheet = "\n" + balance_sheet.to_string()

    return balance_sheet


# get stock info and recommendations summary
def yf_get_stockinfo(
    ticker: str = Field(description="the ticker/trading symbol of the company"),
) -> str:
    """Provides the detailed financial and general information about the stock ticker.
        Also provides a table for analyst recommendations for the ticker.

    Args:
        ticker (str): The ticker symbol of the company to fetch information for.

    Returns:
        str: A string containing the stock info and a summary of recommendations.
    """

    if "." in ticker:
        ticker = ticker.split(".")[0]

    ticker += ".NS"

    company = yf.Ticker(ticker)

    stock_info = company.info
    try:
        recommendations_summary = company.recommendations_summary
    except Exception as e:
        print(f"No recommendations {e}")
        recommendations_summary = ""

    # TODO: add units and convert to easily understandable units

    include_info = [
        "industry",
        "sector",
        "longBuisnessSummary",
        "previousClose",
        "dividendRate",
        "dividentYield",
        "beta",
        "forwardPE",
        "volume",
        "marketCap",
        "fiftyTwoWeekLow",
        "fiftyTwoWeekHigh",
        "currency",
        "bookValue",
        "priceToBook",
        "earningsQuarterlyGrowth",
        "trailingEps",
        "forwardEps",
        "52WeekChange",
        "totalCashPerShare",
        "ebidta",
        "totabDebt",
        "totalCashPerShare",
        "debtToEquity",
        "revenuePerShare",
        "earningsGrowth",
        "revenueGrowth",
        "grossMargins",
        "ebidtaMargins",
        "operatingMargins",
    ]
    response = "## Stock info:\n"

    for key, val in stock_info.items():
        if key in include_info:
            if re.search("(Growth|Margin|Change)", key):
                response += f"{key}: {round(float(val) * 100, 3)} %\n"
            elif "marketCap" in key:
                response += f"{key}: {round(int(val) * 1e-7, 2)} Cr.\n"
            else:
                response += f"{key}: {val}\n"

    response += "\n## Analyst Recommendations:\n"
    response += f"\n{recommendations_summary}"

    return response


def yf_fundamental_analysis(
    ticker: str = Field(description="the ticker/trading symbol of the company"),
):
    """
    Perform a comprehensive fundamental analysis on the given stock symbol.

    Args:
        stock_symbol (str): The stock symbol to analyze.

    Returns:
        dict: A dictionary with the detailed fundamental analysis results.
    """
    try:
        if "." in ticker:
            ticker = ticker.split(".")[0]
        # add the NS tag to fetch correct data (NSE India).
        ticker += ".NS"

        stock = yf.Ticker(ticker)
        info = stock.info

        # Data processing
        financials = stock.financials.infer_objects(copy=False)
        balance_sheet = stock.balance_sheet.infer_objects(copy=False)
        cash_flow = stock.cashflow.infer_objects(copy=False)

        # Fill missing values
        financials = financials.ffill()
        balance_sheet = balance_sheet.ffill()
        cash_flow = cash_flow.ffill()

        # Key Ratios and Metrics
        ratios = {
            "P/E Ratio": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "P/B Ratio": info.get("priceToBook"),
            "P/S Ratio": info.get("priceToSalesTrailing12Months"),
            "PEG Ratio": info.get("pegRatio"),
            "Debt to Equity": info.get("debtToEquity"),
            "Current Ratio": info.get("currentRatio"),
            "Quick Ratio": info.get("quickRatio"),
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "ROIC": info.get("returnOnCapital"),
            "Gross Margin": info.get("grossMargins"),
            "Operating Margin": info.get("operatingMargins"),
            "Net Profit Margin": info.get("profitMargins"),
            "Dividend Yield": info.get("dividendYield"),
            "Payout Ratio": info.get("payoutRatio"),
        }

        # Growth Rates
        revenue = financials.loc["Total Revenue"] if not financials.empty else []
        net_income = financials.loc["Net Income"] if not financials.empty else []
        revenue_growth = (
            revenue.pct_change(periods=-1).iloc[0] if len(revenue) > 1 else None
        )
        net_income_growth = (
            net_income.pct_change(periods=-1).iloc[0] if len(net_income) > 1 else None
        )

        growth_rates = {
            "Revenue Growth (YoY)": revenue_growth,
            "Net Income Growth (YoY)": net_income_growth,
        }

        # Valuation
        market_cap = info.get("marketCap")
        enterprise_value = info.get("enterpriseValue")

        valuation = {
            "Market Cap": market_cap,
            "Enterprise Value": enterprise_value,
            "EV/EBITDA": info.get("enterpriseToEbitda"),
            "EV/Revenue": info.get("enterpriseToRevenue"),
        }

        # Future Estimates
        estimates = {
            "Next Year EPS Estimate": info.get("forwardEps"),
            "Next Year Revenue Estimate": info.get("revenueEstimates", {}).get("avg"),
            "Long-term Growth Rate": info.get("longTermPotentialGrowthRate"),
        }

        # Simple DCF Valuation (very basic)
        free_cash_flow = (
            cash_flow.loc["Free Cash Flow"].iloc[0]
            if "Free Cash Flow" in cash_flow.index
            else None
        )
        wacc = 0.1  # Assumed Weighted Average Cost of Capital
        growth_rate = info.get("longTermPotentialGrowthRate", 0.03)

        def simple_dcf(fcf, growth_rate, wacc, years=5):
            if fcf is None or growth_rate is None:
                return None
            terminal_value = fcf * (1 + growth_rate) / (wacc - growth_rate)
            dcf_value = sum(
                [
                    fcf * (1 + growth_rate) ** i / (1 + wacc) ** i
                    for i in range(1, years + 1)
                ]
            )
            dcf_value += terminal_value / (1 + wacc) ** years
            return dcf_value

        dcf_value = simple_dcf(free_cash_flow, growth_rate, wacc)

        # Prepare the results
        analysis = {
            "Company Name": info.get("longName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Key Ratios": ratios,
            "Growth Rates": growth_rates,
            "Valuation Metrics": valuation,
            "Future Estimates": estimates,
            "Simple DCF Valuation": dcf_value,
            "Last Updated": dt.fromtimestamp(
                info.get("lastFiscalYearEnd", 0)
            ).strftime("%Y-%m-%d"),
            "Data Retrieval Date": dt.now().strftime("%Y-%m-%d"),
        }

        # Add interpretations
        interpretations = {
            "P/E Ratio": (
                "High P/E might indicate overvaluation or high growth expectations"
                if ratios.get("P/E Ratio", 0) > 16
                else "Low P/E might indicate undervaluation or low growth expectations"
            ),
            "Debt to Equity": (
                "High leverage"
                if ratios.get("Debt to Equity", 0) > 2
                else "Conservative capital structure"
            ),
            "ROE": (
                "Couldn't find ROE"
                if not ratios.get("ROE", 0.0)
                else (
                    "Strong returns"
                    if ratios.get("ROE", 0.0) > 0.15
                    else "Potential profitability issues"
                )
            ),
            "Revenue Growth": (
                "Couldn't get Revenue Growth"
                if not growth_rates.get("Revenue Growth (YoY)")
                else (
                    "Strong growth"
                    if growth_rates.get("Revenue Growth (YoY)", 0) > 0.1
                    else "Slowing growth"
                )
            ),
        }

        analysis["Interpretations"] = interpretations

        return analysis

    except Exception as e:
        return f"An error occurred during the analysis: {e}"


def analyse_company_yf(
    company_name: str = Field(
        description="The name of the company, accurately extracted from the query",
        pattern=r"""^\w[\w.\-#&\s]*$""",
    ),
) -> str:
    """Perform analysis on the company specified in the input query.

    This function takes a query about a company,
    finds its ticker symbol, retrieves financial statements and stock information,
    Args:
        company_name (str): The input query containing the company name to analyze.

    Returns:
        str: A string containing the financial statements and stock information
             of the specified company."""

    try:
        # Check if the company is on the NSE
        company = CompanyName(company_name=company_name)
        # get the ticker
        company_symbol = stock_search(company.company_name)
        ticker = Ticker(company_symbol=company_symbol)
        print(ticker)
        # get fundamental analysis, financials & info
        fundamental_analysis = yf_fundamental_analysis(ticker.company_symbol)
        financials = yf_get_financial_statements(ticker.company_symbol)
        info = yf_get_stockinfo(ticker.company_symbol)

        # get recent news
        news = get_recent_news(ticker.company_symbol)

        return "\n".join([json.dumps(fundamental_analysis), info, financials, news])

    except ValidationError:
        return f"Not a valid company/ticker -> {company_name}: {company_symbol}"

    except Exception as e:
        return f"Error fetching data, Please try again: {e}"
