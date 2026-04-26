"""Utilities to fetch EU regulation text from EUR-Lex via the Publications Office SPARQL API.

The fetcher resolves a CELEX identifier to a work URI, finds an XHTML manifestation
for a target language, then downloads the XHTML payload as text.
"""

import logging
import time

import httpx

from src.ingestion.config import REGULATIONS, SPARQL_ENDPOINT
from src.ingestion.models import FetchResult

logger = logging.getLogger(__name__)


class EurLexFetcher:
    """Fetches regulation XHTML from EUR-Lex for a configured language.

    Args:
        language: Three-letter EU language code used in the expression lookup
            (for example: "FRA", "ENG").
    """

    def __init__(self, language: str = "FRA"):
        self.language = language

    async def _get_work_uri(self, celex: str, client: httpx.AsyncClient) -> str:
        """Resolve a CELEX identifier to the corresponding EUR-Lex work URI.

        Raises:
            httpx.HTTPStatusError: If the SPARQL endpoint responds with an error.
            ValueError: If no work is found for the provided CELEX number.
        """
        logger.debug("Resolving work URI for celex=%s", celex)
        query = f"""
        SELECT ?work WHERE {{
            ?work <http://publications.europa.eu/ontology/cdm#resource_legal_id_celex>
            "{celex}"^^<http://www.w3.org/2001/XMLSchema#string> .
        }} LIMIT 1
        """
        response = await client.post(
            SPARQL_ENDPOINT,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
        )
        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]
        if not bindings:
            logger.warning("No work URI found for celex=%s", celex)
            raise ValueError(f"No work found for CELEX {celex}")
        return bindings[0]["work"]["value"]

    async def _get_xhtml_uri(self, work_uri: str, client: httpx.AsyncClient) -> str:
        """Find an XHTML manifestation URI for the configured language and work.

        Raises:
            httpx.HTTPStatusError: If the SPARQL endpoint responds with an error.
            ValueError: If no matching manifestations or XHTML type is found.
        """
        logger.debug("Looking up manifestations for work_uri=%s language=%s", work_uri, self.language)
        query = f"SELECT ?manifestation ?type WHERE {{ ?expression <http://publications.europa.eu/ontology/cdm#expression_belongs_to_work> <{work_uri}> . ?expression <http://publications.europa.eu/ontology/cdm#expression_uses_language> <http://publications.europa.eu/resource/authority/language/{self.language}> . ?manifestation <http://publications.europa.eu/ontology/cdm#manifestation_manifests_expression> ?expression . ?manifestation <http://publications.europa.eu/ontology/cdm#manifestation_type> ?type . }} LIMIT 10"
        response = await client.get(
            SPARQL_ENDPOINT,
            params={"query": query},
            headers={"Accept": "application/sparql-results+json"},
        )
        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]
        if not bindings:
            logger.warning("No manifestations found for work_uri=%s language=%s", work_uri, self.language)
            raise ValueError(f"No manifestations found for {work_uri}")
        for binding in bindings:
            if binding["type"]["value"] == "xhtml":
                return binding["manifestation"]["value"]
        logger.warning("No xhtml manifestation found for work_uri=%s language=%s", work_uri, self.language)
        raise ValueError(f"No XHTML manifestation found for {work_uri}")

    async def _download_xhtml(self, xhtml_uri: str, client: httpx.AsyncClient) -> str:
        """Download the XHTML content from a manifestation URI.

        Raises:
            httpx.HTTPStatusError: If the resource request fails.
        """
        logger.debug("Downloading xhtml from uri=%s", xhtml_uri)
        response = await client.get(
            xhtml_uri,
            headers={"Accept": "application/xhtml+xml"},
        )
        response.raise_for_status()
        return response.text

    async def fetch(self, regulation: str) -> FetchResult:
        """Fetch XHTML for a supported regulation alias and return normalized metadata."""
        meta = REGULATIONS.get(regulation.lower())
        if not meta:
            logger.error("Unknown regulation alias=%s", regulation)
            raise ValueError(f"Unknown regulation: {regulation}. Choose from {list(REGULATIONS.keys())}")

        celex = meta["celex"]
        start = time.perf_counter()
        logger.info("Fetch started regulation=%s celex=%s language=%s", regulation, celex, self.language)

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                work_uri = await self._get_work_uri(celex, client)
                logger.debug("Resolved work URI celex=%s work_uri=%s", celex, work_uri)
                xhtml_uri = await self._get_xhtml_uri(work_uri, client)
                logger.debug("Resolved XHTML URI celex=%s xhtml_uri=%s", celex, xhtml_uri)
                html = await self._download_xhtml(xhtml_uri, client)
        except Exception:
            logger.exception("Fetch failed regulation=%s celex=%s", regulation, celex)
            raise

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "Fetch completed regulation=%s celex=%s html_chars=%s duration_ms=%s",
            regulation,
            celex,
            len(html),
            elapsed_ms,
        )

        return FetchResult(
            html=html,
            regulation_name=meta["regulation_name"],
            valid_from=meta["valid_from"],
            source_url=f"https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:{celex}",
        )
