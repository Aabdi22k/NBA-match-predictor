import pandas as pd

logs = pd.read_csv('logs_base.csv')
games = pd.read_csv('nba_schedule.csv')

logs.drop(columns=['SEASON_YEAR', 'TEAM_ID', 'TEAM_NAME', 'GAME_ID', 'AVAILABLE_FLAG', 
       'PTS',], inplace=True)

logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])

tricodes = logs['TEAM_ABBREVIATION'].unique()

results = []

cols = ['MIN', 'FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA','FT_PCT','OREB','DREB','REB','AST','TOV','STL','BLK','BLKA','PF','PFD', 'PLUS_MINUS']
new_cols = [f'{c}_rolling' for c in cols]

for code in tricodes:
    data = logs[logs['TEAM_ABBREVIATION'] == code]
    data = data.sort_values(by='GAME_DATE', ascending=False)
    stats = data.head(5).mean(numeric_only=True)
    # Rename each stat to stat_rolling
    stats = stats.rename(lambda x: f"{x}_rolling", axis=0)
    stats['TEAM_ABBREVIATION'] = code
    results.append(stats)

logs = pd.DataFrame(results)
logs = pd.merge(logs, games, on='TEAM_ABBREVIATION' )

logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
logs['DAY_CODE'] = logs['GAME_DATE'].dt.dayofweek
logs['VENUE'] = 1
teams = {
    0: 'SAS', 1: 'MIA', 2: 'UTA', 3: 'WAS', 4: 'HOU', 5: 'CHI', 6: 'MIN',
    7: 'LAL', 8: 'PHX', 9: 'MIL', 10: 'BKN', 11: 'ORL', 12: 'DAL', 13: 'CHA',
    14: 'ATL', 15: 'GSW', 16: 'SAC', 17: 'IND', 18: 'OKC', 19: 'MEM',
    20: 'POR', 21: 'LAC', 22: 'BOS', 23: 'NOP', 24: 'TOR', 25: 'DEN',
    26: 'DET', 27: 'NYK', 28: 'CLE', 29: 'PHI'
}

reversed_teams = {v: k for k, v in teams.items()}
logs['OPPONENT_CODE'] = logs['OPPONENT'].map(reversed_teams)

logs.to_csv('future_logs.csv', index=False)

