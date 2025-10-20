import tensorflow as tf
import argparse
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import platform
import sys
import logging
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralNetworkBuilder:
    def __init__(self, verbose=False):
        self.model = None
        self.gpu_available = self.check_gpu(verbose)
        self.verbose = verbose
        self.history = None

    def check_gpu(self, verbose=False):
        """Детальная проверка GPU и CUDA"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                if verbose:
                    logger.info(f"Обнаружены GPU: {[gpu.name for gpu in gpus]}")
                    logger.info(f"Версия TensorFlow: {tf.__version__}")
                    cuda_version = tf.sysconfig.get_build_info().get('cuda_version', 'Unknown')
                    cudnn_version = tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown')
                    logger.info(f"CUDA: {cuda_version}, cuDNN: {cudnn_version}")
                return True
            else:
                logger.warning("GPU не обнаружены. Проверьте:")
                logger.warning("1. CUDA 12.4: ls /usr/local/cuda-12.4/lib64/libcudart*")
                logger.warning("2. cuDNN 9.0+: ls /usr/local/cuda-12.4/lib64/libcudnn*")
                logger.warning("3. LD_LIBRARY_PATH: echo $LD_LIBRARY_PATH")
                logger.warning("4. TensorFlow: tensorflow[and-cuda]==2.17.0")
                logger.warning("5. Для Debian 13: установите libtinfo5 из http://deb.debian.org/debian/pool/main/n/ncurses/libtinfo5_6.4-4_amd64.deb")
                logger.warning("См. https://www.tensorflow.org/install/gpu")
                return False
        except Exception as e:
            logger.error(f"Ошибка проверки GPU: {e}")
            return False

    def test_gpu(self):
        """Тест производительности GPU"""
        try:
            with tf.device('/GPU:0'):
                a = tf.random.normal([10000, 10000])
                b = tf.random.normal([10000, 10000])
                start_time = tf.timestamp()
                _ = tf.matmul(a, b)
                elapsed = tf.timestamp() - start_time
                logger.info(f"Тест GPU: матричный расчёт 10000x10000 завершён за {elapsed:.3f} сек")
        except Exception as e:
            logger.error(f"Ошибка теста GPU: {e}")

    def load_data(self, dataset='mnist', custom_path=None):
        """Загрузка данных: MNIST или .npz"""
        try:
            if dataset.lower() == 'mnist':
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                x_train, x_test = x_train / 255.0, x_test / 255.0
                x_train = np.expand_dims(x_train, -1)
                x_test = np.expand_dims(x_test, -1)
                logger.info(f"MNIST загружен: train={x_train.shape}, test={x_test.shape}")
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
                logger.info(f"Кастомный датасет: train={x_train.shape}, test={x_test.shape}")
                return (x_train, y_train), (x_test, y_test)
            else:
                raise ValueError(f"Датасет {dataset} или путь {custom_path} не поддерживаются")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            raise

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
            logger.info(f"Модель {architecture} создана с конфигом: {layers_config}")
            if self.verbose:
                model.summary()
            return model
        except Exception as e:
            logger.error(f"Ошибка создания модели: {e}")
            raise

    def train_model(self, train_data, epochs=5, batch_size=32, validation_data=None):
        """Обучение с прогрессом и графиками"""
        try:
            callbacks = [tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)]
            
            with tqdm(total=epochs, desc="Обучение эпох") as pbar:
                class TqdmCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        pbar.update(1)
                        pbar.set_postfix({'acc': logs.get('accuracy', 0), 'val_acc': logs.get('val_accuracy', 0)})
                
                self.history = self.model.fit(
                    train_data[0], train_data[1],
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    verbose=1 if self.verbose else 0,
                    callbacks=callbacks + [TqdmCallback()]
                )

            # Графики
            if self.history:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(self.history.history['accuracy'], label='Точность (train)')
                plt.plot(self.history.history.get('val_accuracy', []), label='Точность (val)')
                plt.title('Точность модели')
                plt.xlabel('Эпоха')
                plt.ylabel('Точность')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(self.history.history['loss'], label='Лосс (train)')
                plt.plot(self.history.history.get('val_loss', []), label='Лосс (val)')
                plt.title('Лосс модели')
                plt.xlabel('Эпоха')
                plt.ylabel('Лосс')
                plt.legend()

                plt.savefig('training_history.png')
                plt.savefig('training_history.pdf')
                plt.close()
                logger.info("Графики сохранены: training_history.png, training_history.pdf")

        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            raise

    def evaluate_model(self, test_data):
        """Оценка модели на тестовом наборе"""
        try:
            if not self.model:
                raise ValueError("Модель не создана")
            loss, *metrics = self.model.evaluate(test_data[0], test_data[1], verbose=0)
            logger.info(f"Оценка на тесте: loss={loss:.4f}, метрики={dict(zip(self.model.metrics_names[1:], metrics))}")
            return loss, metrics
        except Exception as e:
            logger.error(f"Ошибка оценки: {e}")
            raise

    def save_model(self, path='model.h5'):
        """Сохранение модели"""
        if self.model:
            self.model.save(path)
            logger.info(f"Модель сохранена в {path}")

    def load_model(self, path='model.h5'):
        """Загрузка модели"""
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Модель загружена из {path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            raise

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Builder v5.0")
        self.builder = NeuralNetworkBuilder(verbose=True)
        self.data = None
        self.create_widgets()

    def create_widgets(self):
        """Расширенный GUI"""
        main_frame = ttk.LabelFrame(self.root, text="Настройки нейросети")
        main_frame.pack(pady=5, padx=10, fill='x')

        # Архитектура
        ttk.Label(main_frame, text="Архитектура:").grid(row=0, column=0, sticky='w')
        self.arch_var = tk.StringVar(value="cnn")
        arch_menu = ttk.OptionMenu(main_frame, self.arch_var, "cnn", "cnn", "rnn", "transformer")
        arch_menu.grid(row=0, column=1, sticky='w', pady=2)

        # Эпохи
        ttk.Label(main_frame, text="Эпохи:").grid(row=1, column=0, sticky='w')
        self.epochs_var = tk.IntVar(value=5)
        ttk.Spinbox(main_frame, from_=1, to=100, textvariable=self.epochs_var, width=5).grid(row=1, column=1, sticky='w', pady=2)

        # Batch size
        ttk.Label(main_frame, text="Batch size:").grid(row=2, column=0, sticky='w')
        self.batch_var = tk.IntVar(value=32)
        ttk.Spinbox(main_frame, from_=16, to=512, textvariable=self.batch_var, width=5).grid(row=2, column=1, sticky='w', pady=2)

        # Активация
        ttk.Label(main_frame, text="Активация:").grid(row=3, column=0, sticky='w')
        self.act_var = tk.StringVar(value="relu")
        act_menu = ttk.OptionMenu(main_frame, self.act_var, "relu", "relu", "sigmoid", "tanh")
        act_menu.grid(row=3, column=1, sticky='w', pady=2)

        # Оптимизатор
        ttk.Label(main_frame, text="Оптимизатор:").grid(row=4, column=0, sticky='w')
        self.opt_var = tk.StringVar(value="adam")
        opt_menu = ttk.OptionMenu(main_frame, self.opt_var, "adam", "adam", "sgd", "rmsprop")
        opt_menu.grid(row=4, column=1, sticky='w', pady=2)

        # Learning rate
        ttk.Label(main_frame, text="Learning rate:").grid(row=5, column=0, sticky='w')
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Spinbox(main_frame, from_=0.0001, to=0.1, textvariable=self.lr_var, increment=0.0001, width=8).grid(row=5, column=1, sticky='w', pady=2)

        # Dropout
        ttk.Label(main_frame, text="Dropout (0-0.5):").grid(row=6, column=0, sticky='w')
        self.dropout_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(main_frame, from_=0.0, to=0.5, textvariable=self.dropout_var, increment=0.1, width=5).grid(row=6, column=1, sticky='w', pady=2)

        # BatchNorm
        self.bn_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="Batch Normalization", variable=self.bn_var).grid(row=7, column=0, columnspan=2, sticky='w', pady=2)

        # Conv Layers
        ttk.Label(main_frame, text="Conv слои (фильтры,kernel,stride,padding):").grid(row=8, column=0, sticky='w')
        self.conv_layers_var = tk.StringVar(value="32,3,1,same;64,3,1,same")
        ttk.Entry(main_frame, textvariable=self.conv_layers_var, width=30).grid(row=8, column=1, sticky='w', pady=2)

        # Dense Layers
        ttk.Label(main_frame, text="Dense слои (юниты):").grid(row=9, column=0, sticky='w')
        self.dense_layers_var = tk.StringVar(value="128,64")
        ttk.Entry(main_frame, textvariable=self.dense_layers_var, width=15).grid(row=9, column=1, sticky='w', pady=2)

        # Loss
        ttk.Label(main_frame, text="Loss функция:").grid(row=10, column=0, sticky='w')
        self.loss_var = tk.StringVar(value="sparse_categorical_crossentropy")
        loss_menu = ttk.OptionMenu(main_frame, self.loss_var, "sparse_categorical_crossentropy", "sparse_categorical_crossentropy", "categorical_crossentropy")
        loss_menu.grid(row=10, column=1, sticky='w', pady=2)

        # Прогресс
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(pady=5, fill='x', padx=10)

        # Кнопки
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Тест GPU", command=self.test_gpu_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Загрузить данные", command=self.load_data_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Создать модель", command=self.create_model_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Обучить модель", command=self.train_model_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Оценить модель", command=self.evaluate_model_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Сохранить модель", command=self.save_model_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Загрузить модель", command=self.load_model_gui).pack(side='left', padx=5)

        # Лог
        self.log_text = tk.Text(self.root, height=12, width=80)
        self.log_text.pack(pady=5, padx=10, fill='both', expand=True)

    def log_message(self, message):
        """Логи в GUI"""
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.root.update()

    def test_gpu_gui(self):
        """Тест GPU в GUI"""
        try:
            self.builder.test_gpu()
            self.log_message("Тест GPU успешно завершён!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Тест GPU не удался: {e}")

    def load_data_gui(self):
        """Загрузка данных"""
        dataset = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")])
        if dataset:
            try:
                self.data = self.builder.load_data(custom_path=dataset)
                self.log_message(f"Загружен кастомный датасет: {dataset}")
            except Exception as e:
                self.data = self.builder.load_data(dataset='mnist')
                self.log_message(f"Ошибка кастомного датасета: {e}. Загружен MNIST")
        else:
            self.data = self.builder.load_data(dataset='mnist')
            self.log_message("Загружен MNIST")

    def create_model_gui(self):
        """Создание модели"""
        try:
            conv_layers = []
            for layer in self.conv_layers_var.get().split(';'):
                if layer.strip():
                    filters, kernel, stride, padding = layer.split(',')
                    conv_layers.append({
                        'filters': int(filters),
                        'kernel': int(kernel),
                        'stride': int(stride),
                        'padding': padding.strip()
                    })
            config = {
                'conv_layers': conv_layers,
                'dense_layers': [int(x) for x in self.dense_layers_var.get().split(',') if x],
                'dropout': self.dropout_var.get(),
                'batch_norm': self.bn_var.get(),
                'activation': self.act_var.get(),
                'optimizer': self.opt_var.get(),
                'lr': self.lr_var.get(),
                'loss': self.loss_var.get(),
                'metrics': ['accuracy']
            }
            self.builder.create_model(self.arch_var.get(), config)
            self.log_message(f"Модель {self.arch_var.get()} создана!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать модель: {e}")

    def train_model_gui(self):
        """Обучение"""
        if not self.data:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        if not self.builder.model:
            messagebox.showwarning("Предупреждение", "Сначала создайте модель!")
            return

        def update_progress(epochs_trained, total_epochs):
            self.progress['value'] = (epochs_trained / total_epochs) * 100
            self.root.update()

        original_fit = self.builder.model.fit
        def wrapped_fit(*args, **kwargs):
            kwargs['epochs'] = self.epochs_var.get()
            kwargs['batch_size'] = self.batch_var.get()
            kwargs['validation_data'] = self.data[1]
            history = original_fit(*args, **kwargs, verbose=1)
            update_progress(kwargs['epochs'], kwargs['epochs'])
            return history
        self.builder.model.fit = wrapped_fit.__get__(self.builder.model)

        try:
            self.builder.train_model(self.data[0], epochs=self.epochs_var.get(), batch_size=self.batch_var.get())
            self.progress['value'] = 100
            messagebox.showinfo("Успех", "Обучение завершено! Графики в training_history.png/pdf")
            self.log_message("Обучение завершено, графики сохранены")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обучения: {e}")

    def evaluate_model_gui(self):
        """Оценка модели"""
        if not self.data:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        if not self.builder.model:
            messagebox.showwarning("Предупреждение", "Сначала создайте модель!")
            return
        try:
            loss, metrics = self.builder.evaluate_model(self.data[1])
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in zip(self.builder.model.metrics_names[1:], metrics)])
            self.log_message(f"Оценка на тесте: loss={loss:.4f}, {metrics_str}")
            messagebox.showinfo("Результат", f"Оценка: loss={loss:.4f}, {metrics_str}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка оценки: {e}")

    def save_model_gui(self):
        """Сохранение модели"""
        if self.builder.model:
            path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("H5 files", "*.h5")])
            if path:
                self.builder.save_model(path)
                self.log_message(f"Сохранено: {path}")

    def load_model_gui(self):
        """Загрузка модели"""
        path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if path:
            self.builder.load_model(path)
            self.log_message(f"Загружено: {path}")

def run_gui():
    """Запуск GUI"""
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()

def run_cli(args):
    """Расширенный CLI"""
    builder = NeuralNetworkBuilder(verbose=args.verbose)
    
    try:
        if args.test_gpu:
            builder.test_gpu()
            return

        if args.load_data:
            data = builder.load_data(custom_path=args.load_data)
        else:
            data = builder.load_data(dataset='mnist')

        conv_layers = []
        if args.conv_layers:
            for layer in args.conv_layers.split(';'):
                if layer.strip():
                    filters, kernel, stride, padding = layer.split(',')
                    conv_layers.append({
                        'filters': int(filters),
                        'kernel': int(kernel),
                        'stride': int(stride),
                        'padding': padding.strip()
                    })

        config = {
            'conv_layers': conv_layers if conv_layers else [{'filters': 32, 'kernel': 3, 'stride': 1, 'padding': 'same'}],
            'dense_layers': [int(x) for x in args.dense_layers.split(',') if x] if args.dense_layers else [128],
            'dropout': args.dropout,
            'batch_norm': args.batch_norm,
            'activation': args.activation,
            'optimizer': args.optimizer,
            'lr': args.lr,
            'loss': args.loss,
            'metrics': ['accuracy']
        }

        model = builder.create_model(args.architecture, config)
        
        if args.train:
            builder.train_model(data[0], epochs=args.epochs, batch_size=args.batch_size, validation_data=data[1])
        
        if args.evaluate:
            loss, metrics = builder.evaluate_model(data[1])
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in zip(builder.model.metrics_names[1:], metrics)])
            logger.info(f"Оценка: loss={loss:.4f}, {metrics_str}")
        
        if args.save:
            builder.save_model(args.save)
        
        if args.load:
            builder.load_model(args.load)

    except Exception as e:
        logger.error(f"Ошибка в CLI: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Neural Network Builder v5.0")
    parser.add_argument('--mode', choices=['gui', 'cli'], default='gui', help="Режим: gui или cli")
    parser.add_argument('--architecture', choices=['cnn', 'rnn', 'transformer'], default='cnn', help="Архитектура")
    parser.add_argument('--conv-layers', type=str, default="32,3,1,same;64,3,1,same", help="Conv слои: filters,kernel,stride,padding;...")
    parser.add_argument('--dense-layers', type=str, default="128,64", help="Dense слои (юниты, через запятую)")
    parser.add_argument('--activation', choices=['relu', 'sigmoid', 'tanh'], default='relu', help="Активация")
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop'], default='adam', help="Оптимизатор")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout (0-0.5)")
    parser.add_argument('--batch-norm', action='store_true', help="Включить BatchNorm")
    parser.add_argument('--loss', choices=['sparse_categorical_crossentropy', 'categorical_crossentropy'], default='sparse_categorical_crossentropy', help="Loss функция")
    parser.add_argument('--epochs', type=int, default=5, help="Эпохи обучения")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--load-data', type=str, help="Путь к .npz датасету")
    parser.add_argument('--train', action='store_true', help="Обучить модель")
    parser.add_argument('--evaluate', action='store_true', help="Оценить модель")
    parser.add_argument('--save', type=str, help="Сохранить модель (.h5)")
    parser.add_argument('--load', type=str, help="Загрузить модель (.h5)")
    parser.add_argument('--test-gpu', action='store_true', help="Тест GPU")
    parser.add_argument('--verbose', action='store_true', help="Подробный вывод")
    
    args = parser.parse_args()

    if args.mode == 'gui':
        run_gui()
    else:
        run_cli(args)

if __name__ == "__main__":
    main()
