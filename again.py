import os
import cv2
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Завантаження попередньо навченої моделі CNN
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Передобробка кадру перед передбаченням
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
    return frame

# Передбачення ознак для кадру
def predict_features(frame):
    frame = preprocess_frame(frame)
    frame = tf.expand_dims(frame, axis=0)
    features = model.predict(frame)
    return features

# Папка з відео
video_folder = 'videos'
output_folder = 'res'

# Кількість відео для обробки
num_videos = 30

# Кількість кадрів для обробки на відео
num_frames = 50

# Зменшення розміру відео
resize_width = 640
resize_height = 480

# Зчитування відео та отримання ознак для кожного кадру
video_count = 0
for filename in os.listdir(video_folder):
    if video_count >= num_videos:
        break

    if filename.endswith(".mp4"):
        video_path = os.path.join(video_folder, filename)
        video = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        step = total_frames // num_frames

        frame_index = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if frame_index % step == 0:
                frames.append(frame)
            frame_index += 1
            if len(frames) >= num_frames:
                break

        # Зменшення розміру відео
        resized_frames = [cv2.resize(frame, (resize_width, resize_height)) for frame in frames]

        video_features = []
        for frame in resized_frames:
            features = predict_features(frame)
            video_features.append(features)

        # Об'єднання ознак для всього відео
        combined_features = np.mean(video_features, axis=0)
        
        # Збереження результатів
        output_filename = os.path.splitext(filename)[0] + '.npy'
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, combined_features)

        video_count += 1

        video.release()

    else:
        continue
