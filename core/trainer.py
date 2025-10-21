import tensorflow as tf
from utils.logger import get_logger
from tqdm import tqdm

class Trainer:
    def __init__(self, model, loss='ctc_loss', optimizer='adam', lr=0.001, metrics=None):
        self.logger = get_logger(__name__)
        self.model = model
        self.loss = loss
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.optimizer.learning_rate = lr
        self.metrics = metrics or []
        self.model.compile(optimizer=self.optimizer, loss=self.ctc_loss, metrics=self.metrics)

    def ctc_loss(self, y_true, y_pred):
        """CTC loss с правильной обработкой"""
        label_length = tf.reduce_sum(tf.cast(y_true != -1, tf.int32), axis=1)
        input_length = tf.ones(tf.shape(y_true)[0], dtype=tf.int32) * (y_pred.shape[1])
        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    def train(self, train_data, test_data, epochs, batch_size):
        """Обучение с прогресс-баром"""
        try:
            self.logger.info(f"Начало обучения: {epochs} эпох, batch_size={batch_size}")
            history = self.model.fit(
                train_data,
                validation_data=test_data,
                epochs=epochs,
                verbose=1,
                callbacks=[tqdm_callback(epochs, desc="Обучение эпох", unit="epoch")]
            )
            return history
        except Exception as e:
            self.logger.error(f"Ошибка обучения: {e}")
            raise

def tqdm_callback(epochs, desc, unit):
    """Callback для tqdm прогресс-бара"""
    class TqdmCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs, desc, unit):
            self.pbar = tqdm(total=epochs, desc=desc, unit=unit)

        def on_epoch_end(self, epoch, logs=None):
            self.pbar.update(1)
            self.pbar.set_postfix(loss=logs.get('loss'), val_loss=logs.get('val_loss'))

        def on_train_end(self, logs=None):
            self.pbar.close()

    return TqdmCallback(epochs, desc, unit)
