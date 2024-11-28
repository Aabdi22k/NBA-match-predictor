import pandas as pd

logs = pd.read_csv("logs_base.csv")
print(logs.columns)

def determine_home_or_away(row):
    if row['TEAM_ABBREVIATION'] in row['MATCHUP']:
        if '@' in row['MATCHUP']:
            return 0
        elif 'vs.' in row['MATCHUP']:
            return 1
    return None

def determine_opponent(row):
    teams = row['MATCHUP'].replace('@', '').replace('vs.', '').split()
    if len(teams) == 2:
        return teams[1] if row['TEAM_ABBREVIATION'] == teams[0] else teams[0]
    return None  # Fallback if the data is not as expected

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("GAME_DATE")
    rolling_stats = group[cols].rolling(10, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])


logs['OPPONENT'] = logs.apply(determine_opponent, axis=1)
mapping = {'NOK': 'NOP', 'NOH': 'NOP', 'SEA': 'OKC', 'NJN': 'BKN'}
logs['OPPONENT'] = [mapping.get(opp, opp) for opp in logs['OPPONENT']]
teams = {
    0: 'SAS', 1: 'MIA', 2: 'UTA', 3: 'WAS', 4: 'HOU', 5: 'CHI', 6: 'MIN',
    7: 'LAL', 8: 'PHX', 9: 'MIL', 10: 'BKN', 11: 'ORL', 12: 'DAL', 13: 'CHA',
    14: 'ATL', 15: 'GSW', 16: 'SAC', 17: 'IND', 18: 'OKC', 19: 'MEM',
    20: 'POR', 21: 'LAC', 22: 'BOS', 23: 'NOP', 24: 'TOR', 25: 'DEN',
    26: 'DET', 27: 'NYK', 28: 'CLE', 29: 'PHI'
}

reversed_teams = {v: k for k, v in teams.items()}
logs['OPPONENT_CODE'] = logs['OPPONENT'].map(reversed_teams)

logs['DAY_CODE'] = logs['GAME_DATE'].dt.dayofweek
logs['VENUE'] = logs.apply(determine_home_or_away, axis=1)
logs['TARGET'] = (logs['WL'] == 'W').astype(int)
logs.drop(columns=['MATCHUP'], inplace=True)



cols = ['MIN', 'FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA','FT_PCT','OREB','DREB','REB','AST','TOV','STL','BLK','BLKA','PF','PFD', 'PLUS_MINUS']
new_cols = [f'{c}_rolling' for c in cols]

logs = logs.groupby("TEAM_ABBREVIATION").apply(lambda x: rolling_averages(x, cols, new_cols))
logs = logs.droplevel('TEAM_ABBREVIATION')
logs.index = range(logs.shape[0])

logs.drop(columns=['SEASON_YEAR', 'TEAM_ID', 'TEAM_NAME', 'GAME_ID', 'AVAILABLE_FLAG', 
       'PTS', 'MIN', 'FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA','FT_PCT','OREB','DREB','REB','AST','TOV','STL','BLK','BLKA','PF','PFD', 'WL', 'PLUS_MINUS'], inplace=True)

logs.to_csv('logs.csv', index=False)