#! /bin/bash
name="asset_update_and_predict"

folder="./functions/${name}"
echo "cmd.sh
.gcloudignore" > ${folder}/.gcloudignore

gcloud functions \
  deploy $name \
  --runtime python310 \
  --trigger-http \
  --region us-central1 \
  --set-env-vars 'PROJECT=bi-forecast' \
  --entry-point main \
  --allow-unauthenticated \
  --source $folder \
  --timeout 540s \
  --memory 1024MB \
  --set-secrets 'DB_USERNAME=DB_USERNAME:latest','DB_PASSWORD=DB_PASSWORD:latest','DB_HOST=DB_HOST:latest','DB_PORT=DB_PORT:latest','DB_DATABASE=DB_DATABASE:latest','ALPHAVANTAGE_API_KEY=ALPHAVANTAGE_API_KEY:latest'