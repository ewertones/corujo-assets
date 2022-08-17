#! /bin/bash

name="asset_forecast"

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
  --memory 1024MB
  --timeout 540s
