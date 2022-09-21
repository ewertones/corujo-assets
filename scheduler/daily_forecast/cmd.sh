#! /bin/bash
name="daily_forecast"

folder="./scheduler/${name}"

gcloud scheduler \
  jobs delete $name \
  --location us-central1 \
  --quiet

gcloud scheduler \
  jobs create http $name \
  --attempt-deadline 540s \
  --location us-central1 \
  --schedule "10 * * * *" \
  --uri "https://us-central1-corujo.cloudfunctions.net/asset_forecast/"
