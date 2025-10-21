import numpy as np
import tensorflow as tf
import os
from PIL import Image
from utils.logger import get_logger
import tensorflow.keras.preprocessing.image as img_aug
from tqdm import tqdm

class DataLoader:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789'
        self.char_to_int = {char: idx for idx, char in enumerate(self.alphabet)}
        self.int_to_char = {idx: char for idx, char in enumerate(self.alphabet)}

    def load_captcha_dataset(self, dir_path, img_size=(128, 32), grayscale=True):
        """Загрузка датасета капчи с аугментацией и прогресс-баром"""
        try:
            x, y = [], []
            valid_chars = set(self.alphabet)
            files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
            for filename in tqdm(files, desc="Loading captcha dataset", unit="file"):
                if not filename.endswith('.png'):
                    continue
                label = filename.replace('.png', '')
                if not (4 <= len(label) <= 8 and all(c in valid_chars for c in label)):
                    self.logger.warning(f"Пропущен файл {filename}: неверная длина или символы")
                    continue
                img_path = os.path.join(dir_path, filename)
                img = Image.open(img_path)
                img = img.resize(img_size)
                if grayscale:
                    img = img.convert('L')
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=-1)
                else:
                    img_array = np.array(img) / 255.0
                # Аугментация
                img_array = img_aug.random_rotation(img_array, rg=20)
                img_array = img_aug.random_shift(img_array, wrg=0.1, hrg=0.1)
                img_array = img_aug.random_shear(img_array, intensity=0.1)
                img_array = tf.image.random_brightness(img_array, max_delta=0.1).numpy()  # Замена random_noise
                x.append(img_array)
                y_int = [self.char_to_int[c] for c in label]
                y.append(y_int)

            if not x:
                raise ValueError("Не найдено подходящих изображений")
            x = np.array(x)
            y = np.array([np.pad(seq, (0, 8 - len(seq)), constant_values=-1) for seq in y])
            split = int(len(x) * 0.8)
            train_data = (x[:split], y[:split])
            test_data = (x[split:], y[split:])
            self.logger.info(f"Капча-датасет загружен: train={x[:split].shape}, test={x[split:].shape}")
            return train_data, test_data
        except Exception as e:
            self.logger.error(f"Ошибка загрузки капча-датасета: {e}")
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
