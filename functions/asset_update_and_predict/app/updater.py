import os
import re
from datetime import datetime, time, timedelta, timezone
from string import Template

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from tensorflow.keras.callbacks import EarlyStopping

from models.models import AssetPredictions, Assets, AssetValues


def compile_and_fit(
    model: tf.keras.models.Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tf.keras.models.Sequential:
    PATIENCE = 10
    val_early_stopping = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, mode="min", restore_best_weights=True
    )
    train_early_stopping = EarlyStopping(
        monitor="loss", patience=PATIENCE, mode="min", restore_best_weights=True
    )

    model.compile(
        loss="mean_absolute_error",
        optimizer="adam",
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    EPOCHS = 50
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[val_early_stopping, train_early_stopping],
    )

    return model


def create_seq(df: pd.DataFrame, seq_len: int) -> np.array:
    df_np = df.values
    data = []

    # Create sequences of length seq_len
    for idx in range(len(df_np) - seq_len + 1):
        data.append(df_np[idx : idx + seq_len])

    return np.array(data)


def split_data(data: pd.DataFrame, train_percent: float) -> list[pd.DataFrame]:
    n_train = int(np.round(train_percent * data.shape[0]))

    x_train = data[:n_train, :, :-1]
    y_train = data[:n_train, :, -1]

    x_val = data[n_train:, :, :-1]
    y_val = data[n_train:, :, -1]

    return [x_train, y_train, x_val, y_val]


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
            df = (
                pd.read_sql(
                    session.query(AssetValues).filter_by(asset_id=asset.id).statement,
                    self.db_engine,
                )
                .sort_values(by="date")
                .reset_index()
            )

        # Convert date column into seconds
        df["timestamp"] = pd.to_datetime(df["date"]).map(pd.Timestamp.timestamp)

        # Drop today if exists and get last date
        tz = timezone(timedelta(hours=0))
        midnight_ts = datetime.combine(
            datetime.today(), time.min, tzinfo=tz
        ).timestamp()
        df = df[df["timestamp"] < midnight_ts]
        last_date = df["date"].iat[-1]

        # Drop unused columns
        df = df[["timestamp", "open", "high", "low", "close"]]

        # Create next_close column, which is close, but lagged one day
        df["next_close"] = df["close"].shift(-1)

        # 80% of data will be used for training, 20% for validation
        TRAIN_PERCENT = 0.8

        # To scale features, the model should not have access to future values,
        # so the normalization is done using only training data.
        train_df = df[: int(TRAIN_PERCENT * len(df.index))]
        train_mean = train_df.mean()
        train_std = train_df.std()

        df_norm = (df - train_mean) / train_std

        SEQ_LEN = 3
        df_norm_seq = create_seq(df_norm, SEQ_LEN)
        df_norm_seq, to_predict = df_norm_seq[:-1], df_norm_seq[-1]

        # Split data
        x_train, y_train, x_val, y_val = split_data(df_norm_seq, TRAIN_PERCENT)

        # Create model
        LSTM_UNITS = 64
        DROPOUT_RATE = 0.025
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.LSTM(
                    units=LSTM_UNITS,
                    input_shape=(x_train.shape[1:]),
                    return_sequences=True,
                    activation="relu",
                    dropout=DROPOUT_RATE,
                    recurrent_dropout=DROPOUT_RATE,
                ),
                tf.keras.layers.LSTM(
                    units=LSTM_UNITS,
                    input_shape=(x_train.shape[1:]),
                    activation="relu",
                    dropout=DROPOUT_RATE,
                    recurrent_dropout=DROPOUT_RATE,
                ),
                tf.keras.layers.Dense(units=1),
            ]
        )

        # Fit model
        trained_model = compile_and_fit(model, x_train, y_train, x_val, y_val)

        # Predict next values
        prediction_column_number = len(df.columns) - 1
        to_predict = np.delete(to_predict, prediction_column_number, axis=1)
        to_predict = np.expand_dims(to_predict, axis=0)
        predictions_norm = trained_model.predict(to_predict)
        next_close = predictions_norm[0][-1]
        next_close = next_close * train_std["next_close"] + train_mean["next_close"]

        next_day = last_date + timedelta(days=1)
        with Session(self.db_engine) as session:
            asset = session.query(Assets).filter_by(symbol=self.symbol).first()

            new_values = {"asset_id": asset.id, "date": next_day, "close": next_close}
            stmt = (
                insert(AssetPredictions)
                .values(**new_values)
                .on_conflict_do_update(
                    constraint="ap_uidx",
                    set_=dict(**new_values, updated_at=func.now()),
                )
            )
            session.execute(stmt)
            session.commit()
