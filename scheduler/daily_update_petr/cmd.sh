#! /bin/bash
name = daily_update_btc

folder = "./scheduler/${name}"

symbol = "PETR3.SAO"
asset = "STOCK"

gcloud scheduler \
  jobs create http $name \
  --location us-central1 \
  --schedule "0 0 * * *" \
  --uri "https://us-central1-corujo.cloudfunctions.net/asset_update/?symbol=${symbol}&asset=${asset}"