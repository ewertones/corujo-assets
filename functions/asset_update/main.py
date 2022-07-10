"""Cloud Function that updates Stock Market data on Big Query"""
from string import Template
import os
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
    df.to_gbq(
        destination_table=f"assets.{symbol}",
        project_id="corujo",
        if_exists="replace",
        location="us-central1",
    )

    return f"{len(df.index)} rows retrieved for {symbol} from {df.index.min()} to {df.index.max()}!"
