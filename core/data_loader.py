import numpy as np
import tensorflow as tf
import os
from utils.logger import get_logger

class DataLoader:
    def __init__(self):
        self.logger = get_logger(__name__)

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
