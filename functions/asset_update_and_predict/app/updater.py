import os
import re
from string import Template
import requests
import pandas as pd
import logging

from sqlalchemy import create_engine, insert, Table, MetaData
from sqlalchemy.orm import Session, mapper
from sqlalchemy.exc import IntegrityError

from models.models import Assets, AssetPredictions, AssetValues


class AssetUpdater:
    def __init__(
        self, name: str, _type: str, symbol: str, currency: str, description: str = None
    ):
        self.name = name
        self._type = _type
        self.symbol = symbol
        self.currency = currency
        self.description = description
        self.db_engine = self.get_engine()

    def __repr__(self):
        return f"AssetUpdater(name={self.name}, type={self._type}, symbol={self.symbol}, currency={self.currency}, description={self.description})"

    @staticmethod
    def get_engine():
        DB_USERNAME = os.getenv("DB_USERNAME")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_DATABASE = os.getenv("DB_DATABASE")
        return create_engine(
            f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
        )

    def insert_into_db_if_not_exists(self):
        asset = Assets(
            name=self.name,
            _type=self._type,
            description=self.description,
            symbol=self.symbol,
            currency=self.currency,
        )
        with Session(self.db_engine) as session:
            if not session.query(Assets).filter_by(symbol=self.symbol).first():
                session.add(asset)
                session.commit()

    def update_historical_data(self):
        options = {
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
                "to_drop": [
                    "1b. open (USD)",
                    "2b. high (USD)",
                    "3b. low (USD)",
                    "4b. close (USD)",
                    "6. market cap (USD)",
                ],
            },
        }

        with Session(self.db_engine) as session:
            asset = session.query(Assets).filter_by(symbol=self.symbol).first()
            asset_values_dates = pd.DataFrame(
                session.query(AssetValues.date).filter_by(asset_id=asset.id).all()
            )

        option = options[asset._type]

        ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

        url = (
            "https://www.alphavantage.co/query"
            f"?function={option['function']}"
            f"&{option['symbol'].substitute(symbol=asset.symbol)}"
            f"&outputsize=full&datatype=json"
            f"&apikey={ALPHAVANTAGE_API_KEY}"
        )
        res = requests.get(url)

        df = pd.DataFrame.from_dict(res.json()[option["key"]], orient="index")

        if to_drop := option.get("to_drop"):
            df = df.drop(to_drop, axis=1)

        df = df.apply(pd.to_numeric, errors="ignore")

        df = df.rename(columns=lambda x: re.sub(r"^\d\w?\. | \(.*", "", x)).rename(
            columns=lambda x: re.sub(" ", "_", x)
        )

        df["date"] = pd.to_datetime(df.index)
        df["asset_id"] = asset.id

        if not asset_values_dates.empty:
            df = df[~df["date"].isin(asset_values_dates["date"])]

        df.to_sql("asset_values", self.db_engine, if_exists="append", index=False)

    def predict_future_values(self):
        with Session(self.db_engine) as session:
            asset = session.query(Assets).filter_by(symbol=self.symbol).first()
            df = pd.read_sql(
                session.query(AssetValues).filter_by(asset_id=asset.id).statement,
                self.db_engine,
            )
