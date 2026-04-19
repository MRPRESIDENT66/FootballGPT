"""Scout Agent — searches players from database and enriches with shared knowledge."""

from tools.player_db import search_players, get_team_roster


def run_scout(criteria: dict, query: str, shared_knowledge: str = "") -> str:
    """Search players using criteria, then attach shared retrieval context."""
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

    if shared_knowledge:
        result += f"\n\n--- Shared Knowledge ---\n{shared_knowledge}"

    return result
