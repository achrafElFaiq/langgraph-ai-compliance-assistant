import logging
import os

# Public SPARQL endpoint for EU Publications Office RDF data.
SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

# Centralized logging defaults (overridable via env vars in production).
LOG_LEVEL = os.getenv("INGESTION_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv(
    "INGESTION_LOG_FORMAT",
    "%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOG_DATE_FORMAT = os.getenv("INGESTION_LOG_DATE_FORMAT", "%Y-%m-%dT%H:%M:%S%z")


def setup_logging() -> None:
    """Configure root logger using settings defined above."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

