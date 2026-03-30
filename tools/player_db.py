"""Player database tool — search, filter, and compare players from local data.

Data source: data/players.json (fetched from API-Football via scripts/fetch_players.py)
"""

import json
import os
from typing import Optional

from langchain_core.tools import tool

from config.settings import settings

_players: list[dict] = []


def _load_players() -> list[dict]:
    global _players
    if not _players:
        path = os.path.join(settings.DATA_DIR, "players.json")
        with open(path, "r", encoding="utf-8") as f:
            _players = json.load(f)
    return _players


@tool
def search_players(
    position: Optional[str] = None,
    max_age: Optional[int] = None,
    min_age: Optional[int] = None,
    league: Optional[str] = None,
    club: Optional[str] = None,
    nationality: Optional[str] = None,
    min_goals: Optional[int] = None,
    name: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Search players by criteria.
    Position values: FW, MF, DF, GK.
    League values: Premier League, La Liga, Bundesliga, Serie A, Ligue 1.
    Returns JSON list of matching players."""
    players = _load_players()
    results = []

    for p in players:
        if position and position.upper() not in p.get("position", "").upper():
            continue
        if max_age and p.get("age") and p["age"] > max_age:
            continue
        if min_age and p.get("age") and p["age"] < min_age:
            continue
        if league and league.lower() not in p.get("league", "").lower():
            continue
        if club and club.lower() not in p.get("club", "").lower():
            continue
        if nationality and nationality.lower() not in p.get("nationality", "").lower():
            continue
        if min_goals and p.get("stats", {}).get("goals", 0) < min_goals:
            continue
        if name and name.lower() not in p.get("name", "").lower():
            continue
        results.append(p)

    results = results[:limit]
    if not results:
        return "No players found matching the criteria."
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def get_player_details(player_name: str) -> str:
    """Get full details for a specific player by name."""
    players = _load_players()
    for p in players:
        if player_name.lower() in p["name"].lower():
            return json.dumps(p, ensure_ascii=False, indent=2)
    return f"Player '{player_name}' not found in database."


@tool
def compare_players(player_name_1: str, player_name_2: str) -> str:
    """Compare two players side by side. Returns their stats for comparison."""
    players = _load_players()
    p1 = p2 = None
    for p in players:
        if player_name_1.lower() in p["name"].lower():
            p1 = p
        if player_name_2.lower() in p["name"].lower():
            p2 = p

    if not p1:
        return f"Player '{player_name_1}' not found."
    if not p2:
        return f"Player '{player_name_2}' not found."

    comparison = {
        "player_1": {"name": p1["name"], "age": p1.get("age"), "club": p1.get("club"),
                      "position": p1.get("position"), "stats": p1.get("stats")},
        "player_2": {"name": p2["name"], "age": p2.get("age"), "club": p2.get("club"),
                      "position": p2.get("position"), "stats": p2.get("stats")},
    }
    return json.dumps(comparison, ensure_ascii=False, indent=2)


@tool
def get_team_roster(club_name: str) -> str:
    """Get all players from a specific club."""
    players = _load_players()
    roster = [
        {"name": p["name"], "age": p.get("age"), "position": p.get("position"),
         "nationality": p.get("nationality"), "stats": p.get("stats")}
        for p in players if club_name.lower() in p.get("club", "").lower()
    ]
    if not roster:
        return f"No players found for club '{club_name}'."
    return json.dumps(roster, ensure_ascii=False, indent=2)
