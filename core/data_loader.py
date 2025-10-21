import numpy as np
import tensorflow as tf
import os
from PIL import Image
from utils.logger import get_logger

class DataLoader:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789'
        self.char_to_int = {char: idx for idx, char in enumerate(self.alphabet)}

    def load_captcha_dataset(self, dir_path, img_size=(128, 32), grayscale=True):
        """Загрузка датасета капчи из папки с [метка].png"""
        try:
            x, y = [], []
            valid_chars = set(self.alphabet)
            for filename in os.listdir(dir_path):
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
                    img_array = np.expand_dims(img_array, axis=-1)  # (H, W, 1)
                else:
                    img_array = np.array(img) / 255.0  # (H, W, 3)
                x.append(img_array)
                y_int = [self.char_to_int[c] for c in label]
                y.append(y_int)

            if not x:
                raise ValueError("Не найдено подходящих изображений")
            x = np.array(x)
            y = np.array([np.pad(seq, (0, 8 - len(seq)), constant_values=-1) for seq in y])  # Паддинг до 8
            split = int(len(x) * 0.8)
            train_data = (x[:split], y[:split])
            test_data = (x[split:], y[split:])
            self.logger.info(f"Капча-датасет загружен: train={x[:split].shape}, test={x[split:].shape}")
            return train_data, test_data
        except Exception as e:
            self.logger.error(f"Ошибка загрузки капча-датасета: {e}")
            raise

    def load_data(self, dataset='mnist', custom_path=None):
        """Загрузка данных: MNIST или .npz"""
        try:
            if dataset.lower() == 'mnist':
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                x_train, x_test = x_train / 255.0, x_test / 255.0
                x_train = np.expand_dims(x_train, -1)
                x_test = np.expand_dims(x_test, -1)
                self.logger.info(f"MNIST загружен: train={x_train.shape}, test={x_test.shape}")
                return (x_train, y_train), (x_test, y_test)
            elif custom_path and os.path.exists(custom_path):
                data = np.load(custom_path)
                required_keys = ['x_train', 'y_train', 'x_test', 'y_test']
                if not all(k in data for k in required_keys):
                    raise ValueError(f".npz должен содержать: {required_keys}")
                x_train, y_train = data['x_train'], data['y_train']
                x_test, y_test = data['x_test'], data['y_test']
                if x_train.shape[0] != y_train.shape[0] or x_test.shape[0] != y_test.shape[0]:
                    raise ValueError("Несоответствие размеров данных и меток")
                self.logger.info(f"Кастомный датасет: train={x_train.shape}, test={x_test.shape}")
                return (x_train, y_train), (x_test, y_test)
            else:
                raise ValueError(f"Датасет {dataset} или путь {custom_path} не поддерживаются")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {e}")
            raise

    def get_alphabet(self):
        """Возвращает алфавит для декодирования"""
        return self.alphabet
