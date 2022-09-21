#! /bin/bash
name="daily_update_btc"

folder="./scheduler/${name}"

symbol="BTC"
asset="CRYPTO"

gcloud scheduler jobs delete $name

gcloud scheduler \
  jobs create http $name \
  --location us-central1 \
  --schedule "0 * * * *" \
  --attempt-deadline 540s \
  --uri "https://us-central1-corujo.cloudfunctions.net/asset_update/?symbol=${symbol}&asset=${asset}"