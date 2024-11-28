import requests
import pandas as pd
import datetime

# URL of the JSON data
url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

# Fetch the JSON data
response = requests.get(url)
data = response.json()

# Extract relevant data from the JSON
game_dates = data.get("leagueSchedule", {}).get("gameDates", [])
game_records = []

# Process each game date and game
for game_date in game_dates:
    date = game_date.get("gameDate", "")
    for game in game_date.get("games", []):
        game_info = {
            "GAME_DATE": datetime.datetime.strptime(game.get("gameDateEst", ""), '%Y-%m-%dT%H:%M:%SZ').date(),
            # 'HT': game.get("homeTeam", {}).get("teamId", ""),
            # 'AT': game.get("awayTeam", {}).get("teamId", ""),
            "TEAM_ABBREVIATION": game.get("homeTeam", {}).get("teamTricode", ""),
            "OPPONENT": game.get("awayTeam", {}).get("teamTricode", ""),
        }
        game_records.append(game_info)

# Convert the data into a DataFrame
df = pd.DataFrame(game_records)

# Save the DataFrame to a CSV file
csv_file = "nba_schedule.csv"
df.to_csv(csv_file, index=False)

print(f"Data has been saved to {csv_file}.")
