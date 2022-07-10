#! /bin/bash

gcloud functions \
  deploy first_function \
  --runtime python39 \
  --trigger-http \
  --region us-central1 \
  --set-env-vars 'PROJECT=bi-forecast' \
  --entry-point main \
  --allow-unauthenticated \
  --source "./functions/first_function/"
