import logging
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
from telegram.ext import filters
import cv2
import numpy as np
import tensorflow as tf
from again import predict_features, num_frames, resize_width, resize_height
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

TOKEN = '6279505846:AAHcmihMJ6-eq8C7yQD3UMUxv7DJ9GVyqLw'
model = tf.keras.models.load_model('popularity_model.h5')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Send me your TikTok video, and I'll let you know if it's popular!")

def get_video_file_id(update):
    if update.message.video is not None:
        return update.message.video.file_id
    else:
        return None

def get_video_file_source(file_id):
    url = f'https://api.telegram.org/bot{TOKEN}/getFile'
    params = {'file_id': file_id}
    response = requests.get(url, params=params).json()
    file_path = response['result']['file_path']
    return file_path

def download_video(file_path):
    url = f'https://api.telegram.org/file/bot{TOKEN}/{file_path}'
    response = requests.get(url)
    filename = 'Video.mp4'
    try:
        with open(filename, 'wb') as file:
            file.write(response.content)
        return 'File downloaded successfully!'
    except:
        return 'There was an error while downloading the file.'

async def process_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_id = get_video_file_id(update)
    if file_id is not None:
        file_path = get_video_file_source(file_id)
        video_path = download_video(file_path)
        if video_path == 'File downloaded successfully!':
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Processing video... Please wait.")
            # Зчитування та обробка нового відео
            new_video = cv2.VideoCapture('Video.mp4')
            new_video_frames = []
            total_frames = int(new_video.get(cv2.CAP_PROP_FRAME_COUNT))
            step = total_frames // num_frames

            frame_index = 0
            while new_video.isOpened():
                ret, frame = new_video.read()
                if not ret:
                    break
                if frame_index % step == 0:
                    new_video_frames.append(frame)
                frame_index += 1
                if len(new_video_frames) >= num_frames:
                    break

            # Зменшення розміру нового відео
            resized_frames = [cv2.resize(frame, (resize_width, resize_height)) for frame in new_video_frames]

            frame_features = []
            for frame in resized_frames:
                features = predict_features(frame)
                frame_features.append(features)
            
            frame_features = np.array(frame_features)
            combined_features = np.concatenate([np.mean(frame_features, axis=0), np.std(frame_features, axis=0)])
            combined_features = np.mean(frame_features, axis=0)            
            combined_features = combined_features[:500]
            
            # Передбачення популярності для нового відео
            popularity_prediction = model.predict([combined_features])
            message = "Predicted popularity: " + str(popularity_prediction)

        else:
            message = "There was an error while downloading the video file."
    else:
        message = "No video file found."
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)

def check_update(self, update):
    return self.filters.check_update(update) or False

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    video_filter = filters._MergedFilter(filters._Video(), filters.ChatType.PRIVATE)
    video_handler = MessageHandler(video_filter, process_video)
    application.add_handler(video_handler)
    application.run_polling()
