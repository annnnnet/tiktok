import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# Завантаження датасету
dataset = pd.read_csv('tiktok_dataset.csv')
print(dataset)

# Видалення всіх стовпців, крім id та playCount
dataset = dataset[['id', 'playCount']]
# Видалення дублікатів
dataset = dataset.drop_duplicates()
print(dataset)

# Видалення записів з кількістю переглядів більше 1 мільйону
dataset = dataset[dataset['playCount'] <= 1000000]
# Визначення engagement_score
dataset['popularity'] = dataset['playCount']
print(dataset['popularity'])
# Створення об'єкту StandardScaler
scaler = MinMaxScaler()

# Застосування масштабування до поля popularity
dataset['popularity'] = scaler.fit_transform(dataset['popularity'].values.reshape(-1, 1))
print(dataset)


# Збереження оновленого датасету
dataset.to_csv('tiktok_dataset_updated.csv', index=False)

# --------------------------------------------------------------------------
# Виведення двох колонок 'id' та 'popularity'
# dataset = pd.read_csv('tiktok_dataset_updated.csv')
# subset = dataset[['id', 'popularity']]
# # Фільтрація датасету за значенням popularity > 50%
# filtered_data = dataset[dataset['popularity'] > 0.1]

# # Виведення колонок 'id' та 'popularity' з відфільтрованого датасету
# subset = filtered_data[['id', 'popularity']]
# print(subset)
