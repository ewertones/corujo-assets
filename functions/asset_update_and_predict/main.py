import logging
import yaml
import functions_framework
from flask import Request
from google.cloud import logging as clogging

from app.updater import AssetUpdater

logging_client = clogging.Client()
logging_client.setup_logging()


def read_config(file: str) -> list[AssetUpdater]:
    with open(file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assets = []
    for asset in config.get("assets"):
        assets.append(
            AssetUpdater(
                name=asset["name"],
                _type=asset["type"],
                symbol=asset["symbol"],
                currency=asset["currency"],
                description=asset.get("description"),
            )
        )

    return assets


@functions_framework.http
def main(request: Request) -> tuple[str, int]:
    logging.info(request)

    logging.info("Reading config.yaml...")
    assets = read_config(file="config.yaml")

    logging.info(f"YAML read. {len(assets)} assets parsed!")
    for asset in assets:
        # logging.info(f"Updating {asset.name} asset...")

        # asset.insert_into_db_if_not_exists()

        # logging.info("Updating historical data...")
        # asset.update_historical_data()

        logging.info("Forecasting future values...")
        asset.predict_future_values()
        break  # TODO: Remove

    return "DONE", 200
