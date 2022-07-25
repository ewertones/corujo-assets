"""Cloud Function that updates Stock Market data on Big Query"""
from string import Template
import os
import re
import requests
import pandas as pd
import functions_framework
from flask import Request, Response

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


@functions_framework.http
def main(request: Request) -> Response:
    """Query Alphavantage API to get stock market data, parses result and upload to Big Query"""
    assets = {
        "STOCK": {
            "function": "TIME_SERIES_DAILY",
            "symbol": Template("symbol=$symbol"),
            "key": "Time Series (Daily)",
            "to_drop": ["5. volume"],
        },
        "FOREX": {
            "function": "FX_DAILY",
            "symbol": Template("from_symbol=$symbol&to_symbol=BRL"),
            "key": "Time Series FX (Daily)",
        },
        "CRYPTO": {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": Template("symbol=$symbol&market=BRL"),
            "key": "Time Series (Digital Currency Daily)",
            "to_drop": [
                "1b. open (USD)",
                "2b. high (USD)",
                "3b. low (USD)",
                "4b. close (USD)",
                "5. volume",
                "6. market cap (USD)",
            ],
        },
    }

    try:
        symbol = request.args["symbol"]
    except KeyError:
        return "You must provide a symbol!", 400

    try:
        asset = assets[request.args["asset"]]
    except KeyError:
        return "Incorrect 'asset'! Choose one of STOCK, FOREX, or CRYPTO.", 400

    output_size = request.args.get("outputsize")
    if not output_size:
        output_size = "full"

    url = (
        "https://www.alphavantage.co/query"
        f"?function={asset['function']}"
        f"&{asset['symbol'].substitute(symbol=symbol)}"
        f"&outputsize={output_size}"
        "&datatype=json"
        f"&apikey={ALPHAVANTAGE_API_KEY}"
    )
    res = requests.get(url)

    df = pd.DataFrame.from_dict(res.json()[asset["key"]], orient="index")

    to_drop = asset.get("to_drop")
    if to_drop:
        df = df.drop(to_drop, axis=1)

    df = df.apply(pd.to_numeric, errors="ignore")

    # Column names must contain only letters, numbers, and underscores
    df = df.rename(columns=lambda x: re.sub(r"^\d\w?\. | \(.*", "", x)).rename(
        columns=lambda x: re.sub(" ", "_", x)
    )

    df["date"] = pd.to_datetime(df.index)

    df["asset"] = request.args["asset"]
    df["symbol"] = request.args["symbol"].replace(".", "_")

    df.to_gbq(
        destination_table=f"assets.{symbol.replace('.', '_')}",
        project_id="corujo",
        if_exists="replace",
        location="us-central1",
        table_schema=[{"name": "date", "type": "DATE"}],
    )

    return f"{len(df.index)} rows retrieved for {symbol} from {df.index.min()} to {df.index.max()}!"
