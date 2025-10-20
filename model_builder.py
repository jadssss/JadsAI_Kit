import tensorflow as tf
from utils.logger import get_logger

class ModelBuilder:
    def __init__(self, verbose=False):
        self.model = None
        self.verbose = verbose
        self.logger = get_logger(__name__)

    def create_model(self, architecture='cnn', layers_config=None, input_shape=(28, 28, 1), num_classes=10):
        """Создание модели с расширенными настройками"""
        if layers_config is None:
            layers_config = {
                'conv_layers': [{'filters': 32, 'kernel': 3, 'stride': 1, 'padding': 'same'}],
                'dense_layers': [128],
                'dropout': 0.2,
                'batch_norm': False,
                'activation': 'relu',
                'optimizer': 'adam',
                'lr': 0.001,
                'loss': 'sparse_categorical_crossentropy',
                'metrics': ['accuracy']
            }

        try:
            model = tf.keras.Sequential()
            optimizer = tf.keras.optimizers.get({
                'class_name': layers_config['optimizer'],
                'config': {'learning_rate': layers_config['lr']}
            })

            if architecture.lower() == 'cnn':
                model.add(tf.keras.layers.Input(shape=input_shape))
                for layer in layers_config.get('conv_layers', [{'filters': 32, 'kernel': 3, 'stride': 1, 'padding': 'same'}]):
                    model.add(tf.keras.layers.Conv2D(
                        filters=layer['filters'],
                        kernel_size=layer['kernel'],
                        strides=layer['stride'],
                        padding=layer['padding'],
                        activation=layers_config['activation']
                    ))
                    if layers_config.get('batch_norm', False):
                        model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
                    if layers_config.get('dropout', 0) > 0:
                        model.add(tf.keras.layers.Dropout(layers_config['dropout']))
                model.add(tf.keras.layers.Flatten())
                for units in layers_config.get('dense_layers', [128]):
                    model.add(tf.keras.layers.Dense(units, activation=layers_config['activation']))
                    if layers_config.get('dropout', 0) > 0:
                        model.add(tf.keras.layers.Dropout(layers_config['dropout']))
                model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

            elif architecture.lower() == 'rnn':
                model.add(tf.keras.layers.Input(shape=(None, input_shape[1] * input_shape[2])))
                model.add(tf.keras.layers.LSTM(64, return_sequences=False))
                if layers_config.get('batch_norm', False):
                    model.add(tf.keras.layers.BatchNormalization())
                for units in layers_config.get('dense_layers', [128]):
                    model.add(tf.keras.layers.Dense(units, activation=layers_config['activation']))
                    if layers_config.get('dropout', 0) > 0:
                        model.add(tf.keras.layers.Dropout(layers_config['dropout']))
                model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

            elif architecture.lower() == 'transformer':
                model.add(tf.keras.layers.Input(shape=input_shape))
                model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=layers_config['activation']))
                model.add(tf.keras.layers.MaxPooling2D((2, 2)))
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(128, activation=layers_config['activation']))
                model.add(tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32))
                if layers_config.get('dropout', 0) > 0:
                    model.add(tf.keras.layers.Dropout(layers_config['dropout']))
                model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

            else:
                raise ValueError(f"Архитектура {architecture} не поддерживается")

            model.compile(optimizer=optimizer, loss=layers_config['loss'], metrics=layers_config['metrics'])
            self.model = model
            self.logger.info(f"Модель {architecture} создана с конфигом: {layers_config}")
            if self.verbose:
                model.summary()
            return model
        except Exception as e:
            self.logger.error(f"Ошибка создания модели: {e}")
            raise

    def save_model(self, path='model.h5'):
        """Сохранение модели"""
        if self.model:
            self.model.save(path)
            self.logger.info(f"Модель сохранена в {path}")

    def load_model(self, path='model.h5'):
        """Загрузка модели"""
        try:
            self.model = tf.keras.models.load_model(path)
            self.logger.info(f"Модель загружена из {path}")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки: {e}")
            raise
