import tensorflow as tf
from tqdm import tqdm
from utils.logger import get_logger
from utils.visualizer import Visualizer

class Trainer:
    def __init__(self, model_builder):
        self.model_builder = model_builder
        self.history = None
        self.logger = get_logger(__name__)
        self.visualizer = Visualizer()

    def train_model(self, train_data, epochs=5, batch_size=32, validation_data=None):
        """Обучение с прогрессом и графиками"""
        try:
            if not self.model_builder.model:
                raise ValueError("Модель не создана")
            
            callbacks = [tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)]
            
            with tqdm(total=epochs, desc="Обучение эпох") as pbar:
                class TqdmCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        pbar.update(1)
                        pbar.set_postfix({'loss': logs.get('loss', 0), 'val_loss': logs.get('val_loss', 0)})
                
                self.history = self.model_builder.model.fit(
                    train_data[0], train_data[1],
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    verbose=1 if self.model_builder.verbose else 0,
                    callbacks=callbacks + [TqdmCallback()]
                )

            self.visualizer.plot_history(self.history)
            self.logger.info("Обучение завершено, графики сохранены: training_history.png/pdf")
        except Exception as e:
            self.logger.error(f"Ошибка обучения: {e}")
            raise

    def evaluate_model(self, test_data):
        """Оценка модели (для CTC loss — только loss)"""
        try:
            if not self.model_builder.model:
                raise ValueError("Модель не создана")
            loss = self.model_builder.model.evaluate(test_data[0], test_data[1], verbose=0)
            self.logger.info(f"Оценка на тесте: loss={loss:.4f}")
            return loss, []
        except Exception as e:
            self.logger.error(f"Ошибка оценки: {e}")
            raise
