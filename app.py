from dotenv import load_dotenv
import pulp
import pandas as pd
import numpy as np
import requests
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
from waitress import serve

load_dotenv()

# Get the current directory
current_directory = os.getcwd()
# Define the path to the directories within the current directory
models_directory = os.path.join(current_directory, "models")
data_directory = os.path.join(current_directory, "data2425")


api_keys = os.environ


app = Flask(__name__)
CORS(app)


def authorization_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        headers = request.headers
        auth = headers.get("Authorization")
        if auth in api_keys.values():
            return func(*args, **kwargs)
        else:
            return jsonify({"message": "ERROR: Unauthorized"}), 401

    return decorated_function


def SolveLP(df, SquadComposition, MaxElementsPerTeam, BudgetLimit):
    # Get a list of players
    players = list(df["web_name"])
    # Initialize Dictionaries for Salaries and Positions
    cost = dict(zip(players, df["now_cost"]))
    positions = dict(zip(players, df["element_type"]))
    teams = dict(zip(players, df["team"]))
    # Dictionary for Projected Score for each player
    project_points = dict(zip(players, df["top_ownership"]))
    # Set Players to Take either 1 or 0 values (owned or not)
    player_vars = pulp.LpVariable.dicts(
        "Player", players, lowBound=0, upBound=1, cat="Integer"
    )

    total_score = pulp.LpProblem("FPL Best Team", pulp.LpMaximize)
    total_score += pulp.lpSum([project_points[i] * player_vars[i] for i in player_vars])
    total_score += (
        pulp.lpSum([cost[i] * player_vars[i] for i in player_vars]) <= BudgetLimit
    )
    # Get indices of players for each position
    fwd = [p for p in positions.keys() if positions[p] == 4]
    mid = [p for p in positions.keys() if positions[p] == 3]
    defD = [p for p in positions.keys() if positions[p] == 2]
    gk = [p for p in positions.keys() if positions[p] == 1]
    # Set Constraints
    total_score += (
        pulp.lpSum([player_vars[i] for i in fwd]) == SquadComposition["Forwards"]
    )
    total_score += (
        pulp.lpSum([player_vars[i] for i in defD]) == SquadComposition["Defenders"]
    )
    total_score += (
        pulp.lpSum([player_vars[i] for i in mid]) == SquadComposition["Midfielders"]
    )
    total_score += (
        pulp.lpSum([player_vars[i] for i in gk]) == SquadComposition["Goalkeepers"]
    )

    # Teams constraints
    for k in list(df["team"].unique()):
        teamTMP = [p for p in teams.keys() if teams[p] == k]
        total_score += (
            pulp.lpSum([player_vars[i] for i in teamTMP]) <= MaxElementsPerTeam
        )

    total_score.solve(pulp.PULP_CBC_CMD(msg=False))

    playersTeam = []
    for v in total_score.variables():
        if v.varValue > 0:
            playersTeam.append(
                v.name.replace("Player_r_", "").replace("_", " ").replace("Player ", "")
            )
        #   print(v.name.replace("Player_r_","").replace("_", " ").replace("Player ",""))

    dfPlayers = pd.DataFrame(playersTeam)
    dfPlayers.columns = ["name"]

    merged_df = df.merge(dfPlayers, left_on="web_name", right_on="name", how="inner")
    merged_df = merged_df.drop(columns=["name"])
    return merged_df


def get_team_data(entry_id, gameweek):
    """Retrieve the gw-by-gw data for a specific entry/team

    credit: vaastav/Fantasy-Premier-League/getters.py

    Args:
        entry_id (int) : ID of the team whose data is to be retrieved
        gameweek (int) : Specific gameweek
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    full_url = base_url + str(entry_id) + "/event/" + str(gameweek) + "/picks/"
    response = requests.get(full_url)
    response.raise_for_status()
    data = response.json()
    team_picks = pd.DataFrame(data["picks"])
    team_picks = team_picks.merge(
        player_data()[
            [
                "id",
                "web_name",
                "now_cost",
                "event_points",
                "element_type",
                "form",
                "selected_by_percent",
                "news",
                "team",
                "photo",
                "preds",
            ]
        ],
        left_on="element",
        right_on="id",
    )
    team_picks["photo"] = team_picks["photo"].str.replace(".jpg", ".png", regex=False)
    return team_picks


def current_gameweek():
    gameweek_data = pd.read_pickle(
        os.path.join(data_directory, "get_gameweek_data.pkl")
    )
    try:
        current = gameweek_data[gameweek_data["is_current"]].iloc[-1]["id"]
    except IndexError:  # catch gameweek 0
        current = gameweek_data[gameweek_data["is_next"]].iloc[-1]["id"] - 1
    return current


def player_data():
    gameweek = current_gameweek()
    players = pd.read_pickle(
        os.path.join(data_directory, f"get_player_data_gw{gameweek}.pkl")
    )
    return players


def fixtures_data():
    fixtures = pd.read_pickle(os.path.join(data_directory, "get_fixtures_data.pkl"))
    return fixtures


def club_data():
    clubs = pd.read_pickle(os.path.join(data_directory, "get_club_data.pkl"))
    return clubs


def top_managers_data():
    gameweek = current_gameweek()
    top_managers = pd.read_pickle(
        os.path.join(data_directory, f"top250_gw{gameweek}.pkl")
    )
    return top_managers


def ai_team_data():
    gameweek = current_gameweek()
    ai = pd.read_pickle(os.path.join(data_directory, f"ai_team_gw{gameweek}.pkl"))
    return ai


def fpl_challenge_data():
    gameweek = current_gameweek()
    ai = pd.read_pickle(os.path.join(data_directory, f"FPL_challenge_gw{gameweek}.pkl"))
    return ai


@app.route("/api/fixtures")
@authorization_required
def fixtures_api():
    fixtures = fixtures_data()
    fixtures["event"] = fixtures["event"].fillna(0)
    fixturesdf = fixtures[
        [
            "code",
            "event",
            "id",
            "team_a",
            "team_h",
            "team_a_difficulty",
            "team_h_difficulty",
            "team_code_a",
            "team_code_h",
            "team_name_a",
            "team_name_h",
            "team_short_name_a",
            "team_short_name_h",
        ]
    ]
    fixturesdf = fixturesdf.replace({np.nan: None})
    return {"fixtures": fixturesdf.to_dict(orient="records")}


@app.route("/api/fpl/<int:team_id>")
@authorization_required
def fpl_team(team_id: int):
    team_data = get_team_data(team_id, gameweek=current_gameweek())
    team_data = team_data.merge(
        club_data()[["team_code", "team_id", "team_name", "team_short_name"]],
        left_on="team",
        right_on="team_id",
    )
    team_data = team_data.replace({np.nan: None})
    return {"my_team": team_data.to_dict(orient="records")}


@app.route("/api/top250")
@authorization_required
def top_FPL_managers():
    top_team = top_managers_data()
    top_team = top_team.merge(
        club_data()[["team_code", "team_id", "team_name", "team_short_name"]],
        left_on="team",
        right_on="team_id",
    )
    top_team = top_team.replace({np.nan: None})
    return {"top250": top_team.to_dict(orient="records")}


@app.route("/api/ai")
@authorization_required
def ai_team():
    ai = ai_team_data()
    ai = ai.merge(
        club_data()[["team_code", "team_id", "team_name", "team_short_name"]],
        left_on="team",
        right_on="team_id",
    )
    ai = ai.replace({np.nan: None})
    return {"ai": ai.to_dict(orient="records")}


@app.route("/api/fpl-challenge")
@authorization_required
def fpl_challenge():
    ai = fpl_challenge_data()
    ai = ai.merge(
        club_data()[["team_code", "team_id", "team_name", "team_short_name"]],
        left_on="team",
        right_on="team_id",
    )
    ai = ai.replace({np.nan: None})
    return {"ai": ai.to_dict(orient="records")}


@app.route("/api/players")
@authorization_required
def players_api():
    all_players = player_data()
    all_players = all_players.merge(
        club_data()[["team_code", "team_id", "team_name", "team_short_name"]],
        left_on="team",
        right_on="team_id",
    )
    all_players = all_players.replace({np.nan: None})
    return {"players": all_players.to_dict(orient="records")}


@app.route("/api/gameweek_number")
def gameweek_number():
    gameweek = current_gameweek()
    return f"{gameweek}"


if __name__ == "__main__":
    serve(app, port="8000")
