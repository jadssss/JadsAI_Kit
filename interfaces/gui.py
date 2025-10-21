import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from core.model_builder import ModelBuilder
from core.data_loader import DataLoader
from core.trainer import Trainer
from core.gpu_checker import GPUChecker
from core.predictor import Predictor
from utils.logger import get_logger

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Builder v5.0")
        self.logger = get_logger(__name__)
        self.model_builder = ModelBuilder(verbose=True)
        self.data_loader = DataLoader()
        self.trainer = Trainer(self.model_builder)
        self.gpu_checker = GPUChecker(verbose=True)
        self.predictor = Predictor(self.model_builder, self.data_loader)
        self.data = None
        self.create_widgets()

    def create_widgets(self):
        """GUI интерфейс"""
        main_frame = ttk.LabelFrame(self.root, text="Настройки нейросети")
        main_frame.pack(pady=5, padx=10, fill='x')

        ttk.Label(main_frame, text="Архитектура:").grid(row=0, column=0, sticky='w')
        self.arch_var = tk.StringVar(value="crnn")
        arch_menu = ttk.OptionMenu(main_frame, self.arch_var, "crnn", "cnn", "rnn", "transformer", "crnn")
        arch_menu.grid(row=0, column=1, sticky='w', pady=2)

        ttk.Label(main_frame, text="Эпохи:").grid(row=1, column=0, sticky='w')
        self.epochs_var = tk.IntVar(value=10)
        ttk.Spinbox(main_frame, from_=1, to=100, textvariable=self.epochs_var, width=5).grid(row=1, column=1, sticky='w', pady=2)

        ttk.Label(main_frame, text="Batch size:").grid(row=2, column=0, sticky='w')
        self.batch_var = tk.IntVar(value=64)
        ttk.Spinbox(main_frame, from_=16, to=512, textvariable=self.batch_var, width=5).grid(row=2, column=1, sticky='w', pady=2)

        ttk.Label(main_frame, text="Активация:").grid(row=3, column=0, sticky='w')
        self.act_var = tk.StringVar(value="relu")
        act_menu = ttk.OptionMenu(main_frame, self.act_var, "relu", "relu", "sigmoid", "tanh")
        act_menu.grid(row=3, column=1, sticky='w', pady=2)

        ttk.Label(main_frame, text="Оптимизатор:").grid(row=4, column=0, sticky='w')
        self.opt_var = tk.StringVar(value="adam")
        opt_menu = ttk.OptionMenu(main_frame, self.opt_var, "adam", "adam", "sgd", "rmsprop")
        opt_menu.grid(row=4, column=1, sticky='w', pady=2)

        ttk.Label(main_frame, text="Learning rate:").grid(row=5, column=0, sticky='w')
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Spinbox(main_frame, from_=0.0001, to=0.1, textvariable=self.lr_var, increment=0.0001, width=8).grid(row=5, column=1, sticky='w', pady=2)

        ttk.Label(main_frame, text="Dropout (0-0.5):").grid(row=6, column=0, sticky='w')
        self.dropout_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(main_frame, from_=0.0, to=0.5, textvariable=self.dropout_var, increment=0.1, width=5).grid(row=6, column=1, sticky='w', pady=2)

        self.bn_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Batch Normalization", variable=self.bn_var).grid(row=7, column=0, columnspan=2, sticky='w', pady=2)

        ttk.Label(main_frame, text="Conv слои (фильтры,kernel,stride,padding):").grid(row=8, column=0, sticky='w')
        self.conv_layers_var = tk.StringVar(value="32,3,1,same;64,3,1,same;128,3,1,same")
        ttk.Entry(main_frame, textvariable=self.conv_layers_var, width=30).grid(row=8, column=1, sticky='w', pady=2)

        ttk.Label(main_frame, text="Dense слои (юниты):").grid(row=9, column=0, sticky='w')
        self.dense_layers_var = tk.StringVar(value="128")
        ttk.Entry(main_frame, textvariable=self.dense_layers_var, width=15).grid(row=9, column=1, sticky='w', pady=2)

        ttk.Label(main_frame, text="Loss функция:").grid(row=10, column=0, sticky='w')
        self.loss_var = tk.StringVar(value="ctc_loss")
        loss_menu = ttk.OptionMenu(main_frame, self.loss_var, "ctc_loss", "sparse_categorical_crossentropy", "categorical_crossentropy", "ctc_loss")
        loss_menu.grid(row=10, column=1, sticky='w', pady=2)

        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(pady=5, fill='x', padx=10)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Тест GPU", command=self.test_gpu_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Загрузить данные", command=self.load_data_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Загрузить капча-датасет", command=self.load_captcha_dataset_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Создать модель", command=self.create_model_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Обучить модель", command=self.train_model_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Оценить модель", command=self.evaluate_model_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Сохранить модель", command=self.save_model_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Загрузить модель", command=self.load_model_gui).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Предсказать капчу", command=self.predict_captcha_gui).pack(side='left', padx=5)

        self.log_text = tk.Text(self.root, height=12, width=80)
        self.log_text.pack(pady=5, padx=10, fill='both', expand=True)

    def log_message(self, message):
        """Логи в GUI"""
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.root.update()

    def test_gpu_gui(self):
        """Тест GPU"""
        try:
            self.gpu_checker.test_gpu()
            self.log_message("Тест GPU успешно завершён!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Тест GPU не удался: {e}")

    def load_data_gui(self):
        """Загрузка данных"""
        dataset = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")])
        if dataset:
            try:
                self.data = self.data_loader.load_data(custom_path=dataset)
                self.log_message(f"Загружен кастомный датасет: {dataset}")
            except Exception as e:
                self.data = self.data_loader.load_data(dataset='mnist')
                self.log_message(f"Ошибка кастомного датасета: {e}. Загружен MNIST")
        else:
            self.data = self.data_loader.load_data(dataset='mnist')
            self.log_message("Загружен MNIST")

    def load_captcha_dataset_gui(self):
        """Загрузка капча-датасета"""
        dir_path = filedialog.askdirectory(title="Выберите папку с [метка].png")
        if dir_path:
            try:
                self.data = self.data_loader.load_captcha_dataset(dir_path)
                self.log_message(f"Капча-датасет загружен из {dir_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить: {e}")

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
                'metrics': ['accuracy'] if self.loss_var.get() != 'ctc_loss' else []
            }
            self.model_builder.create_model(self.arch_var.get(), config, input_shape=(32, 128, 1), num_classes=70)
            self.log_message(f"Модель {self.arch_var.get()} создана!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать модель: {e}")

    def train_model_gui(self):
        """Обучение"""
        if not self.data:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        if not self.model_builder.model:
            messagebox.showwarning("Предупреждение", "Сначала создайте модель!")
            return

        def update_progress(epochs_trained, total_epochs):
            self.progress['value'] = (epochs_trained / total_epochs) * 100
            self.root.update()

        original_fit = self.model_builder.model.fit
        def wrapped_fit(*args, **kwargs):
            kwargs['epochs'] = self.epochs_var.get()
            kwargs['batch_size'] = self.batch_var.get()
            kwargs['validation_data'] = self.data[1]
            history = original_fit(*args, **kwargs, verbose=1)
            update_progress(kwargs['epochs'], kwargs['epochs'])
            return history
        self.model_builder.model.fit = wrapped_fit.__get__(self.model_builder.model)

        try:
            self.trainer.train_model(self.data[0], epochs=self.epochs_var.get(), batch_size=self.batch_var.get())
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
        if not self.model_builder.model:
            messagebox.showwarning("Предупреждение", "Сначала создайте модель!")
            return
        try:
            loss, _ = self.trainer.evaluate_model(self.data[1])
            self.log_message(f"Оценка на тесте: loss={loss:.4f}")
            messagebox.showinfo("Результат", f"Оценка: loss={loss:.4f}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка оценки: {e}")

    def save_model_gui(self):
        """Сохранение модели"""
        if self.model_builder.model:
            path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("H5 files", "*.h5")])
            if path:
                self.model_builder.save_model(path)
                self.log_message(f"Сохранено: {path}")

    def load_model_gui(self):
        """Загрузка модели"""
        path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if path:
            self.model_builder.load_model(path)
            self.log_message(f"Загружено: {path}")

    def predict_captcha_gui(self):
        """Предсказание капчи"""
        img_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if img_path:
            try:
                prediction = self.predictor.predict(img_path)
                self.log_message(f"Предсказание для {img_path}: {prediction}")
                messagebox.showinfo("Результат", f"Капча: {prediction}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка предсказания: {e}")
