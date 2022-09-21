#! /bin/bash
name="daily_update_usd"

folder="./scheduler/${name}"

symbol="USD"
asset="FOREX"

gcloud scheduler \
  jobs create http $name \
  --location us-central1 \
  --schedule "0 * * * *" \
  --timeout 540s \
  --uri "https://us-central1-corujo.cloudfunctions.net/asset_update/?symbol=${symbol}&asset=${asset}"