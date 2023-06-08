import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import glob
import cv2

# Функція для завантаження ознак кадрів з файлів .npy
def load_features(folder_path):
    file_paths = glob.glob(folder_path + '/*.npy')
    features = []
    for path in file_paths:
        video_features = np.load(path)
        features.append(video_features)
    features = np.concatenate(features)
    return features

# Завантаження ознак кадрів
folder_path = 'res'
features = load_features(folder_path)

# Завантаження міток популярності з файлу .csv
labels = pd.read_csv('tiktok_dataset_updated.csv', nrows=30)['popularity']

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Побудова моделі нейронної мережі
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))  # Змінено input_shape
model.add(Dense(1, activation='linear'))

# Компіляція моделі
model.compile(loss='mse', optimizer='adam')

# Тренування моделі
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Оцінка моделі на тестовій вибірці
score = model.evaluate(X_test, y_test, verbose=0)
print("Mean Squared Error:", score)

# Збереження навченої моделі
model_filename = 'popularity_model.h5'
model.save(model_filename)
