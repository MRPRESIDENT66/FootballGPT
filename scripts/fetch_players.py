"""Incremental player data fetcher from API-Football.

Usage:
    python scripts/fetch_players.py                # fetch all remaining teams
    python scripts/fetch_players.py --teams 5      # fetch next 5 teams only

Supports incremental fetching:
  - Tracks which teams are already fetched
  - Appends new data to existing players.json
  - Run daily to gradually build up the database

Free tier: 100 req/day, 10 req/min, seasons 2022-2024.
"""

import argparse
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

BASE_URL = "https://v3.football.api-sports.io"
SEASON = 2024

# All teams with hard-coded IDs (no lookup needed)
ALL_TEAMS = [
    # Premier League — Day 1 priority
    {"id": 42, "name": "Arsenal", "league": "Premier League"},
    {"id": 50, "name": "Manchester City", "league": "Premier League"},
    {"id": 40, "name": "Liverpool", "league": "Premier League"},
    {"id": 49, "name": "Chelsea", "league": "Premier League"},
    {"id": 33, "name": "Manchester United", "league": "Premier League"},
    {"id": 47, "name": "Tottenham", "league": "Premier League"},
    {"id": 34, "name": "Newcastle", "league": "Premier League"},
    {"id": 66, "name": "Aston Villa", "league": "Premier League"},
    # La Liga
    {"id": 541, "name": "Real Madrid", "league": "La Liga"},
    {"id": 529, "name": "Barcelona", "league": "La Liga"},
    {"id": 530, "name": "Atletico Madrid", "league": "La Liga"},
    {"id": 531, "name": "Athletic Bilbao", "league": "La Liga"},
    # Bundesliga
    {"id": 157, "name": "Bayern Munich", "league": "Bundesliga"},
    {"id": 168, "name": "Bayer Leverkusen", "league": "Bundesliga"},
    {"id": 165, "name": "Borussia Dortmund", "league": "Bundesliga"},
    # Serie A
    {"id": 505, "name": "Inter Milan", "league": "Serie A"},
    {"id": 489, "name": "AC Milan", "league": "Serie A"},
    {"id": 496, "name": "Juventus", "league": "Serie A"},
    {"id": 492, "name": "Napoli", "league": "Serie A"},
    # Ligue 1
    {"id": 85, "name": "Paris Saint Germain", "league": "Ligue 1"},
    # Day 2+ extras
    {"id": 46, "name": "Leicester", "league": "Premier League"},
    {"id": 48, "name": "West Ham", "league": "Premier League"},
    {"id": 536, "name": "Sevilla", "league": "La Liga"},
    {"id": 548, "name": "Real Sociedad", "league": "La Liga"},
    {"id": 169, "name": "Eintracht Frankfurt", "league": "Bundesliga"},
    {"id": 160, "name": "RB Leipzig", "league": "Bundesliga"},
    {"id": 487, "name": "Lazio", "league": "Serie A"},
    {"id": 497, "name": "Roma", "league": "Serie A"},
    {"id": 80, "name": "Lyon", "league": "Ligue 1"},
    {"id": 81, "name": "Marseille", "league": "Ligue 1"},
    # Day 3 — more top clubs
    {"id": 35, "name": "Bournemouth", "league": "Premier League"},
    {"id": 51, "name": "Brighton", "league": "Premier League"},
    {"id": 39, "name": "Wolves", "league": "Premier League"},
    {"id": 45, "name": "Everton", "league": "Premier League"},
    {"id": 55, "name": "Brentford", "league": "Premier League"},
    {"id": 63, "name": "Fulham", "league": "Premier League"},
    {"id": 532, "name": "Valencia", "league": "La Liga"},
    {"id": 533, "name": "Villarreal", "league": "La Liga"},
    {"id": 543, "name": "Real Betis", "league": "La Liga"},
    {"id": 162, "name": "Werder Bremen", "league": "Bundesliga"},
    {"id": 163, "name": "Borussia Monchengladbach", "league": "Bundesliga"},
    {"id": 172, "name": "VfB Stuttgart", "league": "Bundesliga"},
    {"id": 502, "name": "Fiorentina", "league": "Serie A"},
    {"id": 499, "name": "Atalanta", "league": "Serie A"},
    {"id": 500, "name": "Bologna", "league": "Serie A"},
    {"id": 79, "name": "Lille", "league": "Ligue 1"},
    {"id": 91, "name": "Monaco", "league": "Ligue 1"},
    {"id": 94, "name": "Rennes", "league": "Ligue 1"},
    # Primeira Liga
    {"id": 212, "name": "FC Porto", "league": "Primeira Liga"},
    {"id": 211, "name": "Benfica", "league": "Primeira Liga"},
    {"id": 228, "name": "Sporting CP", "league": "Primeira Liga"},
    # Eredivisie
    {"id": 194, "name": "Ajax", "league": "Eredivisie"},
    {"id": 197, "name": "PSV Eindhoven", "league": "Eredivisie"},
    {"id": 203, "name": "Feyenoord", "league": "Eredivisie"},
]

request_count = 0
DELAY = 7  # 7s = safe under 10 req/min


def api_get(endpoint: str, params: dict) -> dict:
    global request_count
    headers = {"x-apisports-key": settings.FOOTBALL_API_KEY}
    url = f"{BASE_URL}/{endpoint}"

    resp = httpx.get(url, headers=headers, params=params, timeout=20)
    request_count += 1

    remaining = resp.headers.get("x-ratelimit-remaining", "?")

    if resp.status_code != 200:
        print(f"    ❌ HTTP {resp.status_code}")
        return {"response": [], "paging": {"total": 0}}

    data = resp.json()
    errors = data.get("errors", {})
    if errors:
        print(f"    ❌ {errors}")
        return {"response": [], "paging": {"total": 0}}

    count = data.get("results", 0)
    print(f"    [#{request_count} | left: {remaining}] {count} results")
    time.sleep(DELAY)
    return data


def fetch_team_players(team_id: int) -> list[dict]:
    all_p = []
    for page in range(1, 4):
        print(f"    📥 Page {page}...")
        data = api_get("players", {"team": team_id, "season": SEASON, "page": page})
        players = data.get("response", [])
        if not players:
            break
        all_p.extend(players)
        if page >= data.get("paging", {}).get("total", 1):
            break
    return all_p


def normalize(raw: dict, league: str) -> dict:
    player = raw.get("player", {})
    stats_list = raw.get("statistics", [])
    s = stats_list[0] if stats_list else {}

    games = s.get("games", {})
    goals_d = s.get("goals", {})
    passes = s.get("passes", {})
    tackles = s.get("tackles", {})
    dribbles = s.get("dribbles", {})
    shots = s.get("shots", {})
    fouls = s.get("fouls", {})
    cards = s.get("cards", {})
    penalty = s.get("penalty", {})
    team = s.get("team", {})
    lg = s.get("league", {})

    apps = games.get("appearences") or 0
    mins = games.get("minutes") or 0
    goals = goals_d.get("total") or 0
    assists = goals_d.get("assists") or 0
    m90 = mins / 90 if mins > 0 else 1

    def d90(v):
        return round((v or 0) / m90, 2) if mins > 0 else 0

    pos_map = {"Attacker": "FW", "Midfielder": "MF", "Defender": "DF", "Goalkeeper": "GK"}

    def parse_num(v):
        if not v:
            return None
        try:
            return int(v.split()[0])
        except:
            return None

    return {
        "id": player.get("id"),
        "name": player.get("name", ""),
        "firstname": player.get("firstname", ""),
        "lastname": player.get("lastname", ""),
        "age": player.get("age"),
        "nationality": player.get("nationality", ""),
        "club": team.get("name", ""),
        "league": lg.get("name") or league,
        "position": pos_map.get(games.get("position", ""), games.get("position", "")),
        "position_full": games.get("position", ""),
        "height_cm": parse_num(player.get("height")),
        "weight_kg": parse_num(player.get("weight")),
        "photo": player.get("photo", ""),
        "stats": {
            "season": SEASON,
            "appearances": apps,
            "starts": games.get("lineups") or 0,
            "minutes_played": mins,
            "goals": goals,
            "assists": assists,
            "goals_per_90": d90(goals),
            "assists_per_90": d90(assists),
            "shots_total": shots.get("total") or 0,
            "shots_on_target": shots.get("on") or 0,
            "shots_per_90": d90(shots.get("total") or 0),
            "pass_accuracy_pct": passes.get("accuracy") or 0,
            "key_passes": passes.get("key") or 0,
            "dribbles_attempts": dribbles.get("attempts") or 0,
            "dribbles_success": dribbles.get("success") or 0,
            "dribbles_per_90": d90(dribbles.get("attempts") or 0),
            "tackles_total": tackles.get("total") or 0,
            "tackles_per_90": d90(tackles.get("total") or 0),
            "interceptions": tackles.get("interceptions") or 0,
            "interceptions_per_90": d90(tackles.get("interceptions") or 0),
            "fouls_drawn": fouls.get("drawn") or 0,
            "fouls_committed": fouls.get("committed") or 0,
            "yellow_cards": cards.get("yellow") or 0,
            "red_cards": cards.get("red") or 0,
            "penalty_scored": penalty.get("scored") or 0,
            "penalty_missed": penalty.get("missed") or 0,
            "rating": games.get("rating"),
        },
    }


def load_existing() -> tuple[list[dict], set[int]]:
    """Load existing players.json and return (players, fetched_team_ids)."""
    path = os.path.join(settings.DATA_DIR, "players.json")
    if not os.path.exists(path):
        return [], set()

    with open(path, "r", encoding="utf-8") as f:
        try:
            players = json.load(f)
        except json.JSONDecodeError:
            return [], set()

    if not players:
        return [], set()

    # Figure out which teams we already have
    fetched_clubs = {p.get("club", "") for p in players}
    fetched_ids = set()
    for t in ALL_TEAMS:
        # Match by checking if the team name is in any fetched club name
        for club in fetched_clubs:
            if t["name"].lower() in club.lower() or club.lower() in t["name"].lower():
                fetched_ids.add(t["id"])
                break

    return players, fetched_ids


def save_players(players: list[dict]):
    # Deduplicate by player ID
    seen = set()
    unique = []
    for p in players:
        pid = p.get("id")
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(p)

    unique.sort(key=lambda x: x["stats"]["appearances"], reverse=True)

    path = os.path.join(settings.DATA_DIR, "players.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    return unique


def main():
    parser = argparse.ArgumentParser(description="Fetch player data from API-Football")
    parser.add_argument("--teams", type=int, default=0, help="Max teams to fetch (0=all remaining)")
    args = parser.parse_args()

    if not settings.FOOTBALL_API_KEY:
        print("❌ Set FOOTBALL_API_KEY in .env first")
        sys.exit(1)

    existing, fetched_ids = load_existing()
    remaining_teams = [t for t in ALL_TEAMS if t["id"] not in fetched_ids]

    if not remaining_teams:
        print("✅ All teams already fetched!")
        print(f"📊 {len(existing)} players in database")
        return

    if args.teams > 0:
        remaining_teams = remaining_teams[:args.teams]

    est_requests = len(remaining_teams) * 2
    est_minutes = len(remaining_teams) * 2 * DELAY // 60

    print(f"⚽ FootballGPT Incremental Data Fetcher")
    print(f"📅 Season: {SEASON}/{SEASON + 1}")
    print(f"📊 Existing: {len(existing)} players from {len(fetched_ids)} teams")
    print(f"🆕 To fetch: {len(remaining_teams)} teams")
    print(f"⏱️  Estimated: ~{est_requests} requests, ~{est_minutes} min")
    print("=" * 50)

    new_players = []
    teams_ok = 0

    for t in remaining_teams:
        print(f"\n📋 {t['name']} ({t['league']})")
        raw = fetch_team_players(t["id"])
        count = 0
        for rp in raw:
            p = normalize(rp, t["league"])
            if p["stats"]["appearances"] > 0:
                new_players.append(p)
                count += 1
        print(f"    ✅ {count} players with appearances")
        teams_ok += 1

    # Merge with existing
    all_players = existing + new_players
    unique = save_players(all_players)

    print("\n" + "=" * 50)
    print(f"✅ Done!")
    print(f"🆕 Added {len(new_players)} new players")
    print(f"📊 Total: {len(unique)} players in database")
    print(f"🏟️  Fetched {teams_ok} new teams this run")
    print(f"📡 {request_count} API requests used")

    # Show remaining
    still_left = len(ALL_TEAMS) - len(fetched_ids) - teams_ok
    if still_left > 0:
        print(f"⏳ {still_left} teams remaining — run again tomorrow!")


if __name__ == "__main__":
    main()
