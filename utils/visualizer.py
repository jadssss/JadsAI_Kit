import matplotlib.pyplot as plt
from utils.logger import get_logger

class Visualizer:
    def __init__(self):
        self.logger = get_logger(__name__)

    def plot_history(self, history):
        """Построение графиков обучения"""
        try:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Точность (train)')
            plt.plot(history.history.get('val_accuracy', []), label='Точность (val)')
            plt.title('Точность модели')
            plt.xlabel('Эпоха')
            plt.ylabel('Точность')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Лосс (train)')
            plt.plot(history.history.get('val_loss', []), label='Лосс (val)')
            plt.title('Лосс модели')
            plt.xlabel('Эпоха')
            plt.ylabel('Лосс')
            plt.legend()

            plt.savefig('training_history.png')
            plt.savefig('training_history.pdf')
            plt.close()
            self.logger.info("Графики сохранены: training_history.png, training_history.pdf")
        except Exception as e:
            self.logger.error(f"Ошибка построения графиков: {e}")
            raise
