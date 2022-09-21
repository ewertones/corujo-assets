#! /bin/bash
name="daily_update_btc"

folder="./scheduler/${name}"

symbol="BTC"
asset="CRYPTO"

gcloud scheduler \
  jobs create http $name \
  --location us-central1 \
  --schedule "0 0 * * *" \
  --timeout 540s \
  --uri "https://us-central1-corujo.cloudfunctions.net/asset_update/?symbol=${symbol}&asset=${asset}"