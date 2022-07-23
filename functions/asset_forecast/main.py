import pandas
import logging
import functions_framework
from google.cloud import bigquery, logging as clogging
from flask import Request, Response


bq_client = bigquery.Client()
log_client = clogging.Client()
log_client.setup_logging()


@functions_framework.http
def main(request: Request) -> Response:
    print(f"Pulling assets tables from BigQuery...")
    tables = bq_client.list_tables(dataset="assets")
    for table in tables:
        print(f"{table.full_table_id}")
        rows = bq_client.list_rows(table)
        df = rows.to_dataframe()
        df.to_excel(f"{table.table_id}.xlsx", index=None)

    return "DONE!", 200
