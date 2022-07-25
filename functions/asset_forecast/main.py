import logging
import functions_framework
import pandas as pd
import numpy as np
import tensorflow as tf
from google.cloud import bigquery, logging as clogging
from flask import Request, Response


bq_client = bigquery.Client()
log_client = clogging.Client()
log_client.setup_logging()


def create_seq(df, seq_len):
    df_np = df.values  # DataFrame to ndarray
    data = []

    # Create sequences of length seq_len
    for idx in range(len(df_np) - seq_len):
        data.append(df_np[idx : idx + seq_len])

    return np.array(data)


def split_data(data, train_percent):
    n_train = int(np.round(train_percent * data.shape[0]))

    x_train = data[:n_train, :, :-1]
    y_train = data[:n_train, :, -1]

    x_val = data[n_train:, :, :-1]
    y_val = data[n_train:, :, -1]

    return [x_train, y_train, x_val, y_val]


def compile_and_fit(model, x_train, y_train, x_val, y_val, patience=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
    )

    return history


@functions_framework.http
def main(request: Request) -> Response:
    print(f"Pulling assets tables from BigQuery...")
    tables = bq_client.list_tables(dataset="assets")
    for table in tables:
        # Read data
        print(f"{table.full_table_id}")
        rows = bq_client.list_rows(table)
        df = rows.to_dataframe()

        # Convert date column into seconds
        df["timestamp"] = pd.to_datetime(df["date"]).map(pd.Timestamp.timestamp)
        df = df.sort_values(by="timestamp")
        last_date = df["date"].iat[-1]

        # Drop date, asset & type columns
        df = df[["timestamp", "open", "high", "low", "close"]]

        # Using past 30 days of data to predict next value
        SEQ_LEN = 30
        to_predict = df.tail(SEQ_LEN).copy()

        # Add "next_close" column, which is "close" but lagged in one day
        df["next_close"] = df["close"].shift(periods=-1)
        df = df.drop(df.tail(1).index)

        # 80% of data will be used for training, 20% for validation
        TRAIN_PERCENT = 0.80

        # To scale features, the model should not have access to future values,
        # so the normalization is done using only training data.
        train_df = df[: int(TRAIN_PERCENT * len(df.index))]
        train_mean = train_df.mean()
        train_std = train_df.std()

        df_norm = (df - train_mean) / train_std
        df_norm_seq = create_seq(df_norm, SEQ_LEN)

        # Split data
        x_train, y_train, x_val, y_val = split_data(df_norm_seq, TRAIN_PERCENT)

        # Create model
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.LSTM(
                    32, input_shape=(x_train.shape[1:]), return_sequences=False
                ),
                tf.keras.layers.Dense(units=SEQ_LEN),
            ]
        )

        # Fit model
        history = compile_and_fit(model, x_train, y_train, x_val, y_val)

        # Predict next close
        to_predict_norm = (to_predict - train_mean.loc[:"close"]) / train_std.loc[
            :"close"
        ]
        x_predict = np.expand_dims(to_predict_norm.values, axis=0)

        predictions_norm = model.predict(x_predict)
        predictions = (
            predictions_norm * train_std["next_close"] + train_mean["next_close"]
        )

        # Update BigQuery
        prediction_df = pd.DataFrame(
            {"date": [last_date], "next_close": [predictions[0, -1]]}
        )
        prediction_df.to_gbq(
            destination_table=f"predictions.{table.table_id}",
            project_id="corujo",
            if_exists="append",
            location="us-central1",
            table_schema=[
                {"name": "date", "type": "DATE"},
                {"name": "next_close", "type": "FLOAT"},
            ],
        )

    return "DONE!", 200
