"""Wikipedia scraper — fetches and chunks player, team, and season articles.

Uses Wikipedia-API for structured section access and MediaWiki search API
for name resolution (player/club names → Wikipedia page titles).
"""

import re
import time
import logging
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass, field

import requests
import wikipediaapi

logger = logging.getLogger(__name__)

_wiki = wikipediaapi.Wikipedia(
    user_agent="FootballGPT/1.0 (https://github.com/footballgpt; contact@example.com)",
    language="en",
)

# Max characters per chunk — keeps embeddings focused
_MAX_CHUNK_LEN = 800

# Sections worth extracting per entity type
_PLAYER_SECTIONS = {
    "career", "club career", "international career", "early life",
    "style of play", "playing style", "honours", "personal life",
    "injury", "injuries", "career statistics",
}

_TEAM_SECTIONS = {
    "history", "early history", "recent history", "modern era",
    "playing style", "style of play", "tactics",
    "youth academy", "academy", "youth development",
    "transfers", "players", "current squad",
    "stadium", "honours", "records",
}

_SEASON_SECTIONS = {
    "season summary", "overview", "league table", "final standings",
    "results", "top scorers", "top goalscorers", "awards",
    "managerial changes", "events",
}

# Major league season pages to scrape
SEASON_PAGES = [
    "2024-25 Premier League",
    "2024-25 La Liga",
    "2024-25 Serie A",
    "2024-25 Bundesliga",
    "2024-25 Ligue 1",
    "2024-25 Primeira Liga",
    "2024-25 Eredivisie",
    "2024-25 UEFA Champions League",
    "2024-25 UEFA Europa League",
]


@dataclass
class WikiChunk:
    """A single chunk of Wikipedia content with metadata for Qdrant storage.

    All entity types share the same schema. Fields not applicable to a
    given entity_type are omitted (not stored) rather than set to empty strings.
    """
    text: str
    name: str                          # player/team/season name
    entity_type: str                   # "player" | "team" | "season"
    section: str                       # e.g. "career", "style_of_play", "history"
    source_url: str = ""
    # Player-specific
    position: str = ""                 # FW, MF, DF, GK
    nationality: str = ""
    club: str = ""
    # Team-specific
    league: str = ""                   # also used by player & season
    country: str = ""
    # Season-specific
    season: str = ""                   # e.g. "2024-25"

    def to_payload(self) -> dict:
        """Convert to Qdrant payload dict, omitting empty fields."""
        payload = {
            "text": self.text,
            "name": self.name,
            "entity_type": self.entity_type,
            "section": self.section,
        }
        # Only include non-empty fields — keeps payload clean
        for key in ["source_url", "position", "nationality", "club",
                     "league", "country", "season"]:
            val = getattr(self, key)
            if val:
                payload[key] = val
        return payload


# ---------------------------------------------------------------------------
# MediaWiki search API — resolves names to page titles
# ---------------------------------------------------------------------------

def _search_wikipedia(query: str, limit: int = 3) -> List[str]:
    """Search Wikipedia and return matching page titles."""
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        },
        headers={"User-Agent": "FootballGPT/1.0"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("query", {}).get("search", [])
    return [r["title"] for r in results]


def _get_page(title: str) -> Optional[wikipediaapi.WikipediaPage]:
    """Get a Wikipedia page, return None if it doesn't exist."""
    page = _wiki.page(title)
    if page.exists():
        return page
    return None


def _resolve_player_page(firstname: str, lastname: str) -> Optional[wikipediaapi.WikipediaPage]:
    """Try multiple strategies to find a footballer's Wikipedia page."""
    # Strategy 1: full name
    full_name = f"{firstname.split()[0]} {lastname.split()[-1]}"  # first token + last token
    page = _get_page(full_name)
    if page:
        return page

    # Strategy 2: with "(footballer)" disambiguation
    page = _get_page(f"{full_name} (footballer)")
    if page:
        return page

    # Strategy 3: search API
    search_results = _search_wikipedia(f"{full_name} footballer")
    for title in search_results:
        if any(kw in title.lower() for kw in [lastname.split()[-1].lower(), "footballer"]):
            page = _get_page(title)
            if page:
                return page

    return None


def _resolve_team_page(club_name: str) -> Optional[wikipediaapi.WikipediaPage]:
    """Try to find a club's Wikipedia page."""
    # Try common football club name patterns first (more specific → less specific)
    for suffix in ["F.C.", "FC", "CF"]:
        page = _get_page(f"{club_name} {suffix}")
        if page:
            return page

    # Try direct lookup (Wikipedia may redirect to the right page)
    page = _get_page(club_name)
    if page:
        # Check if it's actually a football club page (not a city/disambiguation)
        summary_lower = (page.summary or "").lower()
        if any(kw in summary_lower for kw in ["football", "soccer", "club", "league"]):
            return page

    # Fallback: search
    search_results = _search_wikipedia(f"{club_name} football club")
    for title in search_results:
        page = _get_page(title)
        if page:
            return page

    return None


# ---------------------------------------------------------------------------
# Section extraction and chunking
# ---------------------------------------------------------------------------

def _normalize_section_name(name: str) -> str:
    """Normalize section name for matching."""
    return name.lower().strip()


def _split_text(text: str, max_len: int = _MAX_CHUNK_LEN) -> List[str]:
    """Split long text into chunks at sentence boundaries."""
    if len(text) <= max_len:
        return [text] if text.strip() else []

    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_len and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _extract_sections(
    page: wikipediaapi.WikipediaPage,
    target_sections: Set[str],
) -> List[Tuple[str, str]]:
    """Extract relevant sections from a Wikipedia page.

    Returns list of (section_name, text) tuples.
    """
    results = []

    def _walk(sections, depth=0):
        for section in sections.values() if isinstance(sections, dict) else sections:
            normalized = _normalize_section_name(section.title)
            # Match if any target is a substring of the section title or vice versa
            matched = any(
                target in normalized or normalized in target
                for target in target_sections
            )
            if matched and section.text.strip():
                results.append((normalized, section.text.strip()))
            # Recurse into subsections
            if section.sections:
                _walk(section.sections, depth + 1)

    _walk(page.sections)

    # If no sections matched, use the summary (intro text)
    if not results and page.summary:
        results.append(("summary", page.summary))

    return results


# ---------------------------------------------------------------------------
# Public API — scrape and chunk
# ---------------------------------------------------------------------------

def scrape_player(
    firstname: str,
    lastname: str,
    position: str = "",
    nationality: str = "",
    league: str = "",
    club: str = "",
) -> List[WikiChunk]:
    """Scrape a player's Wikipedia page and return chunks."""
    page = _resolve_player_page(firstname, lastname)
    if not page:
        logger.info(f"Wikipedia page not found for player: {firstname} {lastname}")
        return []

    display_name = f"{firstname.split()[0]} {lastname.split()[-1]}"
    source_url = page.fullurl
    sections = _extract_sections(page, _PLAYER_SECTIONS)

    # Always include summary as a chunk
    if page.summary and not any(s == "summary" for s, _ in sections):
        sections.insert(0, ("summary", page.summary))

    chunks = []
    for section_name, text in sections:
        section_key = section_name.replace(" ", "_")
        for piece in _split_text(text):
            chunks.append(WikiChunk(
                text=f"{display_name}: {piece}",
                name=display_name,
                entity_type="player",
                section=section_key,
                position=position,
                nationality=nationality,
                club=club,
                league=league,
                source_url=source_url,
            ))

    logger.info(f"Scraped player {display_name}: {len(chunks)} chunks")
    return chunks


def scrape_team(
    club_name: str,
    league: str = "",
    country: str = "",
) -> List[WikiChunk]:
    """Scrape a team's Wikipedia page and return chunks."""
    page = _resolve_team_page(club_name)
    if not page:
        logger.info(f"Wikipedia page not found for team: {club_name}")
        return []

    source_url = page.fullurl
    sections = _extract_sections(page, _TEAM_SECTIONS)

    if page.summary and not any(s == "summary" for s, _ in sections):
        sections.insert(0, ("summary", page.summary))

    # Try to infer country from summary if not provided
    if not country and page.summary:
        summary_lower = page.summary.lower()
        for c in ["english", "spanish", "italian", "german", "french",
                   "portuguese", "dutch", "scottish"]:
            if c in summary_lower:
                country = c.capitalize()
                break

    chunks = []
    for section_name, text in sections:
        section_key = section_name.replace(" ", "_")
        for piece in _split_text(text):
            chunks.append(WikiChunk(
                text=f"{club_name}: {piece}",
                name=club_name,
                entity_type="team",
                section=section_key,
                league=league,
                country=country,
                source_url=source_url,
            ))

    logger.info(f"Scraped team {club_name}: {len(chunks)} chunks")
    return chunks


def scrape_season(page_title: str) -> List[WikiChunk]:
    """Scrape a season page (e.g. '2024-25 Premier League') and return chunks."""
    page = _get_page(page_title)
    if not page:
        logger.info(f"Wikipedia page not found: {page_title}")
        return []

    # Extract season and league from title
    season_match = re.match(r"^(\d{4}[-–]\d{2,4})\s*(.+)$", page_title)
    season_year = season_match.group(1) if season_match else ""
    league = season_match.group(2).strip() if season_match else page_title
    source_url = page.fullurl

    sections = _extract_sections(page, _SEASON_SECTIONS)

    if page.summary and not any(s == "summary" for s, _ in sections):
        sections.insert(0, ("summary", page.summary))

    chunks = []
    for section_name, text in sections:
        section_key = section_name.replace(" ", "_")
        for piece in _split_text(text):
            chunks.append(WikiChunk(
                text=f"{page_title}: {piece}",
                name=page_title,
                entity_type="season",
                section=section_key,
                league=league,
                season=season_year,
                source_url=source_url,
            ))

    logger.info(f"Scraped season {page_title}: {len(chunks)} chunks")
    return chunks


def scrape_all_seasons() -> List[WikiChunk]:
    """Scrape all predefined season pages."""
    all_chunks = []
    for title in SEASON_PAGES:
        chunks = scrape_season(title)
        all_chunks.extend(chunks)
        time.sleep(0.5)  # rate limiting
    return all_chunks
