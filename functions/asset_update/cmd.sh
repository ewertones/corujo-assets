#! /bin/bash
name = "asset_update"

folder = "./functions/${name}"
echo "cmd.sh" > ${folder}/.gcloudignore

gcloud functions \
  deploy $name \
  --runtime python39 \
  --trigger-http \
  --region us-central1 \
  --set-env-vars 'PROJECT=bi-forecast' \
  --entry-point main \
  --allow-unauthenticated \
  --source $folder
