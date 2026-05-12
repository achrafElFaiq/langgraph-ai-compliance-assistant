"""Parse EUR-Lex XHTML into normalized `Article` records.

This parser walks the HTML document in order, tracks current title/chapter context,
and extracts article metadata and body text for downstream ingestion.
"""

import logging
from typing import Optional

from bs4 import BeautifulSoup

from src.domain.models.models import Article
from src.infrastructure.utils import roman_to_int

logger = logging.getLogger(__name__)


class EurLexParser():
    """Parser for EUR-Lex XHTML regulation pages."""

    def parse_html(self, html: str, regulation_name: str, valid_from: str, source_url: str) -> list[Article]:
        """Walk the XHTML document and return one Article per article block, tracking title/chapter context."""

        logger.info("Parse started regulation=%s", regulation_name)
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.find_all(True)
        logger.debug("HTML parsed regulation=%s tag_count=%s", regulation_name, len(tags))

        articles = []
        current_title = None
        current_chapter = None

        for tag in tags:
            tag_classes = tag.get("class")

            if tag.name == "p" and tag_classes and "oj-ti-section-1" in tag_classes:
                text = tag.get_text(strip=True)
                if text.startswith("TITRE"):
                    roman = text.split("TITRE")[1].strip().split()[0]
                    current_title = roman_to_int(roman)
                    current_chapter = None
                elif text.startswith("CHAPITRE"):
                    chapter_num = text.split("CHAPITRE")[1].strip().split()[0]
                    current_chapter = roman_to_int(chapter_num)

            if tag.name == "div" and tag_classes == ["eli-subdivision"] and tag.get("id", "").startswith("art_"):
                try:
                    article = self._parse_article(tag, current_title, current_chapter, regulation_name, valid_from,
                                                  source_url)
                except (TypeError, ValueError):
                    logger.warning("Skipping malformed article block regulation=%s id=%s", regulation_name,
                                   tag.get("id"), exc_info=True)
                    continue
                articles.append(article)

        logger.info("Parse completed regulation=%s article_count=%s", regulation_name, len(articles))
        return articles

    def _extract_article_number(self, article_div) -> int:
        """Read numeric article id from element id like `art_16`."""
        article_id = article_div.get("id")  # "art_16"
        return int(article_id.split("_")[1])

    def _extract_article_title(self, article_div) -> Optional[str]:
        """Return article heading text when present."""
        title_div = article_div.find("div", class_="eli-title")
        if not title_div:
            return None
        return title_div.get_text(strip=True)

    def _extract_content(self, article_div) -> str:
        """Join article body paragraphs into a single text field."""
        paragraphs = article_div.find_all("p", class_="oj-normal")
        return " ".join(p.get_text(strip=True) for p in paragraphs)

    def _parse_article(
            self,
            article_div,
            title_num: Optional[int],
            chapter_num: Optional[int],
            regulation_name: str,
            valid_from: str,
            source_url: str,
    ) -> Article:
        """Build an `Article` model from one article block and current section context."""
        article_number = self._extract_article_number(article_div)
        article_title = self._extract_article_title(article_div)
        content = self._extract_content(article_div)

        breadcrumb_parts = [regulation_name]
        if title_num:
            breadcrumb_parts.append(f"Titre {title_num}")
        if chapter_num:
            breadcrumb_parts.append(f"Chapitre {chapter_num}")
        breadcrumb_parts.append(f"Article {article_number}")
        breadcrumb = " > ".join(breadcrumb_parts)

        return Article(
            regulation_name=regulation_name,
            title_number=title_num,
            chapter_number=chapter_num,
            article_number=article_number,
            article_title=article_title,
            breadcrumb=breadcrumb,
            content=content,
            valid_from=valid_from,
            valid_until=None,
            source_url=source_url,
        )
