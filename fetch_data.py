import io
from joblib import load
import time
import numpy as np
import json
import pandas as pd
import requests
import pulp
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning, module="pulp")


# Get the current directory
current_directory = os.getcwd()
# Define the path to the directories within the current directory
models_directory = os.path.join(current_directory, 'models')
data_directory = os.path.join(current_directory, 'data')


# Overall FPL league ID, 314 for 2019/20 season.
overallLeagueID = 314
# overall league
overall_league_url = "https://fantasy.premierleague.com/api/leagues-classic/"+str(overallLeagueID)+"/standings/"


# Fetch the game data once and use it across the application
try:

    lods_gk = load(os.path.join(models_directory, 'gk_model.joblib'))
    lods_def = load(os.path.join(models_directory, 'def_model.joblib'))
    lods_mid = load(os.path.join(models_directory, 'mid_model.joblib'))
    lods_fwd = load(os.path.join(models_directory, 'fwd_model.joblib'))

    game_data = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
except requests.exceptions.RequestException as e:
    print(e)
    game_data = None


def SolveLP(df, SquadComposition, MaxElementsPerTeam, BudgetLimit, feature:str):
    # Get a list of players
    players = list(df['id'])
    # Initialize Dictionaries for Salaries and Positions
    cost = dict(zip(players, df['now_cost']))
    positions = dict(zip(players, df['element_type']))
    teams=dict(zip(players, df['team']))
    # Dictionary for Projected Score for each player
    project_points = dict(zip(players, df[feature]))
    # Set Players to Take either 1 or 0 values (owned or not)
    player_vars = pulp.LpVariable.dicts("Player", players, lowBound=0, upBound=1, cat='Integer')

    total_score = pulp.LpProblem("FPL Best Team", pulp.LpMaximize)
    total_score += pulp.lpSum([project_points[i] * player_vars[i] for i in player_vars])
    total_score += pulp.lpSum([cost[i] * player_vars[i] for i in player_vars]) <= BudgetLimit
    # Get indices of players for each position
    fwd = [p for p in positions.keys() if positions[p] == 4]
    mid = [p for p in positions.keys() if positions[p] == 3]
    defD = [p for p in positions.keys() if positions[p] == 2]
    gk = [p for p in positions.keys() if positions[p] == 1]
    # Set Constraints
    total_score += pulp.lpSum([player_vars[i] for i in fwd]) == SquadComposition["Forwards"]
    total_score += pulp.lpSum([player_vars[i] for i in mid]) == SquadComposition["Midfielders"]
    total_score += pulp.lpSum([player_vars[i] for i in defD]) == SquadComposition["Defenders"]
    total_score += pulp.lpSum([player_vars[i] for i in gk]) == SquadComposition["Goalkeepers"]


    # Teams constraints
    for k in list(df["team"].unique()):
        teamTMP=[p for p in teams.keys() if teams[p] == k]
        total_score += pulp.lpSum([player_vars[i] for i in teamTMP]) <= MaxElementsPerTeam

    total_score.solve(pulp.PULP_CBC_CMD(msg=False))

    playersTeam=[]
    for v in total_score.variables():
        if v.varValue > 0:
            playersTeam.append(v.name.replace("Player_r_","").replace("_", " ").replace("Player ",""))
            # print(v.name.replace("Player_r_","").replace("_", " ").replace("Player ",""))

    dfPlayers=pd.DataFrame(playersTeam)
    dfPlayers.columns=["player_id"]
    dfPlayers["player_id"] = dfPlayers["player_id"].astype(int)

    merged_df = df.merge(dfPlayers, left_on='id', right_on='player_id', how='inner')
    merged_df = merged_df.drop(columns=['player_id'])
    return merged_df


def load_players():
    gameweek = get_current_gameweek()
    players_df = pd.read_pickle(os.path.join(data_directory, f'get_player_data_gw{gameweek}.pkl'))
    return players_df


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
        load_players()[
            ["id", "web_name", "now_cost", "event_points", "element_type", "form", "selected_by_percent", "news", "team", "photo", "preds"]
        ],
        left_on="element",
        right_on="id",
    )
    return team_picks


def get_game_data():
    """Retrieve the gw-by-gw data

    credit: vaastav/Fantasy-Premier-League/getters.py

    """
    return game_data


def get_gameweek_data():
    gw_data =  pd.DataFrame(get_game_data()["events"])
    gw_data.to_pickle(os.path.join(data_directory, 'get_gameweek_data.pkl'))
    print(f"Successfully fetched gameweek data on: {time.ctime()}")
    return gw_data


def get_player_data():
    gameweek = get_current_gameweek()
    gw_df = pd.DataFrame(get_game_data()["elements"])
    # Trial
    # Define the conditions for applying the models
    condition_1 = (gw_df['element_type'] == 1)
    condition_2 = (gw_df['element_type'] == 2)
    condition_3 = (gw_df['element_type'] == 3)
    condition_4 = (gw_df['element_type'] == 4)

    # Apply model_1 to rows where 'element_type' is 1
    gw_df['preds'] = 0  # Initialize the 'preds' column with default values
    gw_df['preds'] = gw_df['preds'].where(~condition_1, np.round(lods_gk.predict(gw_df[['ep_this','form','value_form','transfers_in_event','clean_sheets','total_points','value_season','bps']])))

    # Apply model_2 to rows where 'element_type' is 2
    gw_df['preds'] = gw_df['preds'].where(~condition_2, np.round(lods_def.predict(gw_df[['ep_this','form','value_form','points_per_game','transfers_in_event','clean_sheets_per_90','total_points']])))

    # Apply model_3 to rows where 'element_type' is 3
    gw_df['preds'] = gw_df['preds'].where(~condition_3, np.round(lods_mid.predict(gw_df[['ep_this','form','value_form','points_per_game','transfers_in_event','expected_goal_involvements','total_points','ict_index','goals_scored']])))

    # Apply model_4 to rows where 'element_type' is 4
    gw_df['preds'] = gw_df['preds'].where(~condition_4, np.round(lods_fwd.predict(gw_df[['ep_this','form','value_form','points_per_game','transfers_in_event','bonus','bps','influence','goals_scored','total_points','expected_goals']])))

    gw_df['photo'] = gw_df['photo'].str.replace('.jpg', '.png', regex=False)

    gw_df.to_pickle(os.path.join(data_directory, f'get_player_data_gw{gameweek}.pkl'))

    print(f"Successfully fetched gw player data and predicted player points on: {time.ctime()}")
    return gw_df


def get_club_data():
    url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/teams.csv"
    s = requests.get(url).content
    teams = pd.read_csv(io.StringIO(s.decode('utf-8')))
    teams.rename(columns={'id':'team_id', 'code':'team_code','name':'team_name','short_name':'team_short_name'}, inplace = True)
    teams.to_pickle(os.path.join(data_directory, 'get_club_data.pkl'))
    print(f"Successfully fetched club data on: {time.ctime()}")
    return teams    


def get_current_gameweek():
    gameweeks = pd.read_pickle(os.path.join(data_directory, 'get_gameweek_data.pkl'))
    try:
        current = gameweeks[gameweeks["is_current"]].iloc[-1]["id"]
    except IndexError:  # catch gameweek 0
        current = gameweeks[gameweeks["is_next"]].iloc[-1]["id"] - 1
    return current


def get_fixtures_data():
    url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/fixtures.csv"
    s = requests.get(url).content
    fixtures = pd.read_csv(io.StringIO(s.decode('utf-8')))
    teams = pd.read_pickle(os.path.join(data_directory, "get_club_data.pkl"))
    combined_df = pd.merge(left=fixtures, right=teams, left_on='team_a', right_on='team_id', how='left', suffixes=('_a', '_h'))
    combined_df = pd.merge(left=combined_df, right=teams, left_on='team_h', right_on='team_id', how='left', suffixes=('_a', '_h'))
    combined_df = combined_df.drop(columns=['team_id_a', 'team_id_h'])
    combined_df.to_pickle(os.path.join(data_directory, 'get_fixtures_data.pkl'))
    print(f"Successfully fetched fixtures on: {time.ctime()}")
    return combined_df


def top_managers():

    gameweek = get_current_gameweek()
    ## Check if local data exists for the current gameweek
    # try:
    #     top250df = pd.read_pickle(f"top250_gw{gameWeek}.pkl")
    # adds the top team ID's to this array
    teamIDarray_all = []

    urls = ["https://fantasy.premierleague.com/api/leagues-classic/314/standings/",
            "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=2",
            "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=3",
            "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=4",
            "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=5"]

    managerParsed_all=[]
    for url in urls:
        response = requests.get(url)
        data = response.text
        parsed = json.loads(data)
        managerParsed_all.append(parsed['standings']['results'])

    final_dataframe_all = pd.DataFrame()

    #get csv of top 15 manager information and write to top_managers.csv
    for page in managerParsed_all:
        for manager in page:
            teamIDarray_all.append(manager['entry'])

    def update_dictionary_key(dictionary, key):
        if key in dictionary:
            dictionary[key] += 1
        else:
            dictionary[key] = 1

    count_dict_all = {}
    count = 0
    print("Fetching Top 250 Managers...")
    ## Progress bar ##
    def print_progress_bar(iteration, total, length=50):
        progress = min(1.0, iteration / total)
        arrow = '#' * int(round(length * progress))
        spaces = ' ' * (length - len(arrow))
        percent = int(progress * 100)
        print(f'\r[{arrow + spaces}] {percent}% Complete', end='', flush=True)
    ## ##

    # for each teamID in the top 15, call the api and update both top_managers_gwInfo.csv and top_managers_gwPicks.csv
    for teamID in teamIDarray_all:
        # Call the get_team_data function
        team_data_all = get_team_data(teamID, gameweek)
        for _, x in team_data_all.iterrows():
            update_dictionary_key(count_dict_all, x['id'])
        # Append the team_data to the final_dataframe
        final_dataframe_all = pd.concat([final_dataframe_all, team_data_all])
        time.sleep(2)
        count+=1
        print_progress_bar(count, 250)
    print()
    # Create a new 'frequency' column using list comprehension
    final_dataframe_all['top_ownership'] = final_dataframe_all['id'].apply((lambda x: count_dict_all.get(x, 0)*0.4))
    
    # Remove duplicate rows
    df_unique = final_dataframe_all.drop_duplicates(subset=['web_name'] ,keep='first')

    df_unique.to_pickle(os.path.join(data_directory, f"all_top250_gw{gameweek}_data.pkl"))
    top250df = SolveLP(df_unique, {"Forwards":3,"Midfielders":5,"Defenders":5, "Goalkeepers": 2}, 3, 1000, 'top_ownership')

    top250df.to_pickle(os.path.join(data_directory, f"top250_gw{gameweek}.pkl"))

    print(f"Successfully fetched top 250 managers for GameWeek {gameweek} on: {time.ctime()}")
    return top250df


def ai_team():
    gameweek = get_current_gameweek()
    player_data = pd.read_pickle(os.path.join(data_directory, f"get_player_data_gw{gameweek}.pkl"))
    ai = SolveLP(player_data, {"Forwards":3,"Midfielders":5,"Defenders":5, "Goalkeepers": 2}, 3, 1000, 'preds')
    ai.to_pickle(os.path.join(data_directory, f"ai_team_gw{gameweek}.pkl"))
    print(f"Successfully created AI team for GameWeek {gameweek} on: {time.ctime()}")
    return ai


def main():
    get_game_data()
    get_gameweek_data()
    get_player_data()
    get_club_data()
    get_fixtures_data()
    top_managers()
    ai_team()


if __name__ == "__main__":
    main()
