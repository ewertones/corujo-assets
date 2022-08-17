#! /bin/bash
name="daily_forecast"

folder="./scheduler/${name}"

gcloud scheduler \
  jobs create http $name \
  --location us-central1 \
  --schedule "0 5 * * *" \
  --uri "https://us-central1-corujo.cloudfunctions.net/asset_forecast/"
