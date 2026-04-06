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


@tool
def get_top_scorers(league: Optional[str] = None, position: Optional[str] = None, limit: int = 10) -> str:
    """Get top scorers ranked by goals. Optionally filter by league and/or position.
    Useful for questions like 'who scores the most in La Liga' or 'top scoring midfielders'."""
    players = _load_players()

    filtered = []
    for p in players:
        if league and league.lower() not in p.get("league", "").lower():
            continue
        if position and position.upper() not in p.get("position", "").upper():
            continue
        goals = p.get("stats", {}).get("goals", 0) or 0
        if goals > 0:
            filtered.append(p)

    filtered.sort(key=lambda p: p.get("stats", {}).get("goals", 0) or 0, reverse=True)
    filtered = filtered[:limit]

    if not filtered:
        return "No players found matching the criteria."

    results = []
    for rank, p in enumerate(filtered, 1):
        s = p.get("stats", {})
        results.append({
            "rank": rank,
            "name": p["name"],
            "club": p.get("club"),
            "league": p.get("league"),
            "position": p.get("position"),
            "age": p.get("age"),
            "goals": s.get("goals", 0),
            "assists": s.get("assists", 0),
            "appearances": s.get("appearances", 0),
            "goals_per_90": s.get("goals_per_90", 0),
        })
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def get_team_stats(club_name: str) -> str:
    """Get aggregated team statistics: average age, total goals/assists, position breakdown,
    top scorer, and squad depth. Useful for tactical analysis and transfer planning."""
    players = _load_players()
    roster = [p for p in players if club_name.lower() in p.get("club", "").lower()]

    if not roster:
        return f"No players found for club '{club_name}'."

    # Position breakdown
    position_counts = {}
    for p in roster:
        pos = p.get("position", "Unknown")
        position_counts[pos] = position_counts.get(pos, 0) + 1

    # Aggregate stats
    ages = [p["age"] for p in roster if p.get("age")]
    total_goals = sum(p.get("stats", {}).get("goals", 0) or 0 for p in roster)
    total_assists = sum(p.get("stats", {}).get("assists", 0) or 0 for p in roster)

    # Top scorer
    top_scorer = max(roster, key=lambda p: p.get("stats", {}).get("goals", 0) or 0)

    # Nationality diversity
    nationalities = set(p.get("nationality", "") for p in roster if p.get("nationality"))

    result = {
        "club": club_name,
        "squad_size": len(roster),
        "average_age": round(sum(ages) / len(ages), 1) if ages else None,
        "position_breakdown": position_counts,
        "total_goals": total_goals,
        "total_assists": total_assists,
        "top_scorer": {
            "name": top_scorer["name"],
            "goals": top_scorer.get("stats", {}).get("goals", 0),
        },
        "nationalities": len(nationalities),
        "nationality_list": sorted(nationalities),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# Stats used for similarity calculation
_SIMILARITY_KEYS = [
    "goals_per_90", "assists_per_90", "shots_per_90", "dribbles_per_90",
    "tackles_per_90", "interceptions_per_90", "pass_accuracy_pct",
]


@tool
def find_similar_players(
    player_name: str,
    position: Optional[str] = None,
    league: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Find players with the most similar statistical profile to a given player.
    Uses per-90-minute stats (goals, assists, shots, dribbles, tackles, interceptions)
    to calculate similarity via Euclidean distance. Optionally filter by position or league."""
    players = _load_players()

    # Find target player
    target = None
    for p in players:
        if player_name.lower() in p.get("name", "").lower():
            target = p
            break

    if not target:
        return f"Player '{player_name}' not found in database."

    target_stats = target.get("stats", {})
    target_vector = [float(target_stats.get(k, 0) or 0) for k in _SIMILARITY_KEYS]

    # Calculate distances
    candidates = []
    for p in players:
        if p["id"] == target["id"]:
            continue
        if position and position.upper() not in p.get("position", "").upper():
            continue
        if league and league.lower() not in p.get("league", "").lower():
            continue

        p_stats = p.get("stats", {})
        p_vector = [float(p_stats.get(k, 0) or 0) for k in _SIMILARITY_KEYS]

        # Euclidean distance
        dist = sum((a - b) ** 2 for a, b in zip(target_vector, p_vector)) ** 0.5
        candidates.append((p, dist))

    candidates.sort(key=lambda x: x[1])
    candidates = candidates[:limit]

    if not candidates:
        return "No similar players found matching the criteria."

    results = {
        "target": {"name": target["name"], "club": target.get("club"),
                    "position": target.get("position")},
        "similar_players": [],
    }
    for p, dist in candidates:
        s = p.get("stats", {})
        results["similar_players"].append({
            "name": p["name"],
            "club": p.get("club"),
            "league": p.get("league"),
            "position": p.get("position"),
            "age": p.get("age"),
            "similarity_score": round(1 / (1 + dist), 3),
            "goals_per_90": s.get("goals_per_90", 0),
            "assists_per_90": s.get("assists_per_90", 0),
            "dribbles_per_90": s.get("dribbles_per_90", 0),
        })
    return json.dumps(results, ensure_ascii=False, indent=2)
