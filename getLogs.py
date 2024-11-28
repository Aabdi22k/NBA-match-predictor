from nba_api.stats.endpoints import teamgamelogs
import pandas as pd
import time 

all_logs = []
for year in range(2025, 2020, -1):
    season = f"{year-1}-{str(year)[-2:]}"  # Convert to 'YYYY-YY' format
    logs = teamgamelogs.TeamGameLogs(season_nullable=season, season_type_nullable='Regular Season')
    time.sleep(1)
    logs = logs.get_data_frames()[0]
    all_logs.append(logs)

df = pd.concat(all_logs, ignore_index=True)
df.to_csv('logs_base.csv', index=False)
