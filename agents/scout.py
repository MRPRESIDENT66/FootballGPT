"""Scout Agent — searches players from database and enriches with Wikipedia knowledge."""

from tools.player_db import search_players, get_player_details, get_team_roster
from knowledge.rag import retrieve_knowledge


def run_scout(criteria: dict, query: str) -> str:
    """Search players using criteria, then enrich top results with Wikipedia background."""
    # Build search args from router-extracted parameters
    search_args = {}
    param_map = {
        "position": "position",
        "max_age": "max_age",
        "min_age": "min_age",
        "league": "league",
        "club": "club",
        "nationality": "nationality",
        "min_pace": "min_pace",
        "max_market_value": "max_market_value",
        "min_goals": "min_goals",
    }
    for key, arg_name in param_map.items():
        val = criteria.get(key)
        if val is not None:
            search_args[arg_name] = val

    result = search_players.invoke(search_args)

    # Also get team roster if a team is mentioned (for recommend intent)
    team = criteria.get("team")
    if team:
        roster = get_team_roster.invoke({"club_name": team})
        result += f"\n\n--- Current {team} roster in database ---\n{roster}"

    # RAG: enrich with Wikipedia background (playing style, career history)
    knowledge = retrieve_knowledge(
        query,
        entity_type="player",
        position=criteria.get("position"),
        league=criteria.get("league"),
        limit=5,
    )
    if knowledge:
        result += f"\n\n--- Wikipedia Background ---\n{knowledge}"

    return result
