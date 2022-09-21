#! /bin/bash
name="daily_update_petr"

folder="./scheduler/${name}"

symbol="PETR3.SAO"
asset="STOCK"

gcloud scheduler \
  jobs create http $name \
  --location us-central1 \
  --schedule "0 * * * *" \
  --attempt-deadline 540s \
  --uri "https://us-central1-corujo.cloudfunctions.net/asset_update/?symbol=${symbol}&asset=${asset}"