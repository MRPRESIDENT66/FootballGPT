"""Ingest pipeline — scrape Wikipedia and load into Qdrant.

Usage:
    python -m knowledge.ingest                    # full ingest (players + teams + seasons)
    python -m knowledge.ingest --only players     # players only
    python -m knowledge.ingest --only teams       # teams only
    python -m knowledge.ingest --only seasons     # seasons only
    python -m knowledge.ingest --limit 10         # only first 10 players/teams (for testing)
"""

import json
import time
import logging
import argparse
from typing import Optional, Tuple, List
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from knowledge.wiki_scraper import scrape_player, scrape_team, scrape_all_seasons
from knowledge.qdrant_store import FootballKnowledgeStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

console = Console()

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "players.json"

# Rate limit: pause between Wikipedia API calls (seconds)
_RATE_LIMIT = 0.5


def _load_entities() -> Tuple[List[dict], List[dict]]:
    """Load unique players and teams from the FM database.

    Returns:
        (players, teams) where each is a list of dicts with relevant fields.
    """
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Deduplicate players by id
    seen_players = set()
    players = []
    for p in raw:
        if p["id"] not in seen_players:
            seen_players.add(p["id"])
            players.append({
                "firstname": p.get("firstname", ""),
                "lastname": p.get("lastname", ""),
                "name": p.get("name", ""),
                "position": p.get("position", ""),
                "nationality": p.get("nationality", ""),
                "league": p.get("league", ""),
                "club": p.get("club", ""),
            })

    # Deduplicate teams by club name
    seen_teams = set()
    teams = []
    for p in raw:
        club = p.get("club", "")
        league = p.get("league", "")
        if club and club not in seen_teams:
            seen_teams.add(club)
            teams.append({"club": club, "league": league})

    return players, teams


def ingest_players(store: FootballKnowledgeStore, limit: Optional[int] = None):
    """Scrape all players from FM database and upsert into Qdrant."""
    players, _ = _load_entities()
    if limit:
        players = players[:limit]

    console.print(f"\n[bold cyan]Scraping {len(players)} players from Wikipedia...[/]")

    all_chunks = []
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Players", total=len(players))

        for p in players:
            firstname = p["firstname"]
            lastname = p["lastname"]
            if not firstname or not lastname:
                skipped += 1
                progress.advance(task)
                continue

            try:
                chunks = scrape_player(
                    firstname=firstname,
                    lastname=lastname,
                    position=p["position"],
                    nationality=p["nationality"],
                    league=p["league"],
                    club=p["club"],
                )
                all_chunks.extend(chunks)
                display = f"{firstname.split()[0]} {lastname.split()[-1]}"
                progress.update(task, description=f"Players: {display} ({len(chunks)} chunks)")
            except Exception as e:
                logger.warning(f"Failed to scrape player {firstname} {lastname}: {e}")
                skipped += 1

            progress.advance(task)
            time.sleep(_RATE_LIMIT)

    console.print(f"  Scraped: {len(all_chunks)} chunks from {len(players) - skipped} players, skipped {skipped}")

    if all_chunks:
        console.print("[bold]Upserting player chunks into Qdrant...[/]")
        store.upsert_chunks(all_chunks)


def ingest_teams(store: FootballKnowledgeStore, limit: Optional[int] = None):
    """Scrape all teams from FM database and upsert into Qdrant."""
    _, teams = _load_entities()
    if limit:
        teams = teams[:limit]

    console.print(f"\n[bold cyan]Scraping {len(teams)} teams from Wikipedia...[/]")

    all_chunks = []
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Teams", total=len(teams))

        for t in teams:
            try:
                chunks = scrape_team(club_name=t["club"], league=t["league"])
                all_chunks.extend(chunks)
                progress.update(task, description=f"Teams: {t['club']} ({len(chunks)} chunks)")
            except Exception as e:
                logger.warning(f"Failed to scrape team {t['club']}: {e}")
                skipped += 1

            progress.advance(task)
            time.sleep(_RATE_LIMIT)

    console.print(f"  Scraped: {len(all_chunks)} chunks from {len(teams) - skipped} teams, skipped {skipped}")

    if all_chunks:
        console.print("[bold]Upserting team chunks into Qdrant...[/]")
        store.upsert_chunks(all_chunks)


def ingest_seasons(store: FootballKnowledgeStore):
    """Scrape season pages and upsert into Qdrant."""
    console.print(f"\n[bold cyan]Scraping season pages from Wikipedia...[/]")

    chunks = scrape_all_seasons()
    console.print(f"  Scraped: {len(chunks)} chunks from season pages")

    if chunks:
        console.print("[bold]Upserting season chunks into Qdrant...[/]")
        store.upsert_chunks(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest Wikipedia football data into Qdrant")
    parser.add_argument("--only", choices=["players", "teams", "seasons"], help="Ingest only this entity type")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of players/teams (for testing)")
    args = parser.parse_args()

    store = FootballKnowledgeStore()

    if args.only is None or args.only == "players":
        ingest_players(store, limit=args.limit)
    if args.only is None or args.only == "teams":
        ingest_teams(store, limit=args.limit)
    if args.only is None or args.only == "seasons":
        ingest_seasons(store)

    # Print summary
    console.print("\n[bold green]Ingest complete![/]")
    table = Table(title="Qdrant Collection Stats")
    table.add_column("Entity Type")
    table.add_column("Chunks", justify="right")
    counts = store.count_by_type()
    for entity_type, count in counts.items():
        table.add_row(entity_type, str(count))
    table.add_row("[bold]Total[/]", f"[bold]{store.count()}[/]")
    console.print(table)


if __name__ == "__main__":
    main()
