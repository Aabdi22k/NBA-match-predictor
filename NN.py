import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, BatchNormalization, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.regularizers import l2



logs = pd.read_csv('logs.csv')

targets = logs['TARGET']

irrelevant_columns = ['TEAM_ABBREVIATION', 'GAME_DATE', 'OPPONENT', 'TARGET']
logs.drop(columns=irrelevant_columns, inplace=True)

scaler = StandardScaler()

feature_columns = logs.columns

logs = scaler.fit_transform(logs)

x_train, x_test, y_train, y_test = train_test_split(logs, targets, test_size=0.2, random_state=42)



# Define the model
model = Sequential([
    Dense(512, activation='relu', input_dim=x_train.shape[1], kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)



# Get predictions on the test set
y_pred = model.predict(x_test)

# Convert predictions to binary (0 or 1)
y_pred_bin = (y_pred > 0.5).astype(int)

# Calculate accuracy on test data
test_accuracy = np.mean(y_pred_bin.flatten() == y_test.values)

# Print test accuracy
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

games = pd.read_csv('future_logs.csv')

abb = games['TEAM_ABBREVIATION']

games.drop(columns=['TEAM_ABBREVIATION', 'GAME_DATE', 'OPPONENT'], inplace=True)
# Scale the features using the previously fitted scaler
game_features = scaler.fit_transform(games)

# Get model predictions
predictions = model.predict(game_features)

# Add predictions to the games DataFrame
games['HOME_WIN_PROB'] = predictions
games['PREDICTION'] = (predictions > 0.5).astype(int)

# Save predictions

games['TEAM_ABBREVIATION'] = abb
games[['TEAM_ABBREVIATION','PREDICTION']].to_csv('game_predictions.csv', index=False)



