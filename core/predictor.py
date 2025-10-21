import tensorflow as tf
import numpy as np
from PIL import Image
from utils.logger import get_logger

class Predictor:
    def __init__(self, model_builder, data_loader):
        self.model_builder = model_builder
        self.data_loader = data_loader
        self.logger = get_logger(__name__)
        self.alphabet = self.data_loader.get_alphabet()
        self.int_to_char = {idx: char for idx, char in enumerate(self.alphabet)}

    def preprocess_image(self, img_path, img_size=(128, 32), grayscale=True):
        """Предобработка изображения для предсказания"""
        try:
            img = Image.open(img_path)
            img = img.resize(img_size)
            if grayscale:
                img = img.convert('L')
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=-1)
            else:
                img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0)  # (1, H, W, C)
        except Exception as e:
            self.logger.error(f"Ошибка обработки изображения {img_path}: {e}")
            raise

    def predict(self, img_path, img_size=(128, 32), grayscale=True):
        """Предсказание строки для капчи"""
        try:
            if not self.model_builder.model:
                raise ValueError("Модель не создана")
            img = self.preprocess_image(img_path, img_size, grayscale)
            pred = self.model_builder.model.predict(img)
            decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=np.ones(1) * pred.shape[1])
            decoded = decoded[0].numpy()[0]
            result = ''.join(self.int_to_char[int(idx)] for idx in decoded if idx != -1)
            self.logger.info(f"Предсказание для {img_path}: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Ошибка предсказания: {e}")
            raise
