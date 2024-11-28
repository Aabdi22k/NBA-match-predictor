import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load the data
# tem = pd.read_csv('TEM.csv')
logs = pd.read_csv('logs.csv')
# schedule = pd.read_csv('nba_schedule.csv')

# # Prepare the dataset for training
# # Convert WL to binary (1 for Win, 0 for Loss)
# logs['WL'] = logs['WL'].apply(lambda x: 1 if x == 'W' else 0)

# # Merge home and away stats
# logs = logs.merge(tem, left_on='TEAM_ID', right_on='TEAM_ID', suffixes=('', '_TEAM'))

# opponent_stats = tem.rename(columns=lambda col: f"OPP_{col}" if col not in ['TEAM_NAME', 'TEAM_ID'] else col)

# logs = logs.merge(opponent_stats, left_on='MATCHUP', right_on='TEAM_ID')
# print(logs.head())

# # Feature Engineering
# features = [
#     'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 'E_PACE',
#     'E_AST_RATIO', 'E_OREB_PCT', 'E_DREB_PCT', 'E_REB_PCT', 
#     'E_TM_TOV_PCT', 'OPP_E_OFF_RATING', 'OPP_E_DEF_RATING', 
#     'OPP_E_NET_RATING', 'OPP_E_PACE', 'OPP_E_AST_RATIO',
#     'OPP_E_OREB_PCT', 'OPP_E_DREB_PCT', 'OPP_E_REB_PCT', 'OPP_E_TM_TOV_PCT'
# ]

targets = logs['TARGET']

irrelevant_columns = ['TEAM_ABBREVIATION', 'GAME_DATE', 'OPPONENT', 'TARGET']
logs.drop(columns=irrelevant_columns, inplace=True)

X = logs
y = targets

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train, y_train)

# Evaluate the model
y_pred = rfr.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_binary))
print("Classification Report:\n", classification_report(y_test, y_pred_binary))

# # Predictions for future games
# schedule = schedule.merge(tem, left_on='HT', right_on='TEAM_ID', suffixes=('_HT', ''))
# schedule = schedule.merge(opponent_stats, left_on='AT', right_on='TEAM_ID', suffixes=('_AT', ''))
# future_games = schedule[features]

# future_games_scaled = scaler.transform(future_games)

games = pd.read_csv('future_logs.csv')

abb = games['TEAM_ABBREVIATION']

games.drop(columns=['TEAM_ABBREVIATION', 'GAME_DATE', 'OPPONENT'], inplace=True)
# Scale the features using the previously fitted scaler
game_features = scaler.fit_transform(games)

preds = rfr.predict(game_features)

games['PREDICTION'] = (preds > 0.5).astype(int)

games['TEAM_ABBREVIATION'] = abb
games[['TEAM_ABBREVIATION','PREDICTION']].to_csv('game_predictions.csv', index=False)
