import numpy as np
import tensorflow as tf
import os
from utils.logger import get_logger
import tensorflow.keras.preprocessing.image as img_aug
from tqdm import tqdm

class DataLoader:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789'
        self.char_to_int = {char: idx for idx, char in enumerate(self.alphabet)}
        self.int_to_char = {idx: char for idx, char in enumerate(self.alphabet)}
        # Создаём таблицу для tf.lookup
        keys = tf.constant(list(self.char_to_int.keys()), dtype=tf.string)
        values = tf.constant(list(self.char_to_int.values()), dtype=tf.int32)
        self.lookup_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values), default_value=-1
        )

    def load_captcha_dataset(self, dir_path, img_size=(128, 32), batch_size=64, grayscale=True):
        """Потоковая загрузка датасета капчи с аугментацией и прогресс-баром"""
        try:
            def parse_image(filename):
                # Извлечение метки из имени файла
                label = tf.strings.regex_replace(tf.strings.split(filename, '/')[-1], r'\.png$', '')
                label = tf.strings.unicode_split(label, input_encoding='UTF-8')
                label = self.lookup_table.lookup(label)  # Преобразуем символы в индексы
                label = tf.pad(label, [[0, 8 - tf.shape(label)[0]]], constant_values=-1)
                # Загрузка и обработка изображения
                img = tf.io.read_file(filename)
                img = tf.image.decode_png(img, channels=1 if grayscale else 3)
                img = tf.image.resize(img, img_size)
                img = img / 255.0
                img = img_aug.random_rotation(img, rg=20)
                img = img_aug.random_shift(img, wrg=0.1, hrg=0.1)
                img = img_aug.random_shear(img, intensity=0.1)
                img = tf.image.random_brightness(img, max_delta=0.1)
                return img, label

            files = tf.data.Dataset.list_files(os.path.join(dir_path, '*.png'))
            dataset_size = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
            self.logger.info(f"Найдено {dataset_size} изображений")

            # Прогресс-бар
            with tqdm(total=dataset_size, desc="Processing dataset", unit="file") as pbar:
                def update_pbar(ds):
                    pbar.update(1)
                    return ds
                dataset = files.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.map(update_pbar)

            # Разделение на train/test
            train_size = int(0.8 * dataset_size)
            train_data = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            test_data = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            self.logger.info(f"Датасет готов: train={train_size}, test={dataset_size - train_size}")
            return train_data, test_data
        except Exception as e:
            self.logger.error(f"Ошибка загрузки датасета: {e}")
            raise

    def load_data(self, dir_path):
        """Загрузка других датасетов"""
        try:
            x, y = [], []
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                img = Image.open(img_path)
                img_array = np.array(img) / 255.0
                x.append(img_array)
                label = filename.split('_')[0]
                y.append(label)
            x = np.array(x)
            y = np.array(y)
            self.logger.info(f"Датасет загружен: x={x.shape}, y={y.shape}")
            return x, y
        except Exception as e:
            self.logger.error(f"Ошибка загрузки датасета: {e}")
            raise

    def get_alphabet(self):
        """Возвращает алфавит и словари"""
        return self.alphabet, self.char_to_int, self.int_to_char
