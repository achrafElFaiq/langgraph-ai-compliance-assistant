import os

# Supported regulation aliases mapped to their CELEX legal identifiers.
REGULATIONS = {
    "mica": {
        "celex": "32023R1114",
        "regulation_name": "MiCA",
        "valid_from": "2024-12-30",
    },
    "dora": {
        "celex": "32022R2554",
        "regulation_name": "DORA",
        "valid_from": "2025-01-17",
    },
    "ai_act": {
        "celex": "32024R1689",
        "regulation_name": "AI Act",
        "valid_from": "2024-08-01",
    },
    "gdpr": {
        "celex": "32016R0679",
        "regulation_name": "GDPR",
        "valid_from": "2018-05-25",
    },
}

# Public SPARQL endpoint for EU Publications Office RDF data.
SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

# Centralized logging defaults (overridable via env vars in production).
LOG_LEVEL = os.getenv("INGESTION_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv(
    "INGESTION_LOG_FORMAT",
    "%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOG_DATE_FORMAT = os.getenv("INGESTION_LOG_DATE_FORMAT", "%Y-%m-%dT%H:%M:%S%z")
