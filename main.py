import argparse
import sys
import tkinter as tk
from interfaces.gui import NeuralNetworkGUI
from interfaces.cli import run_cli

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
        root = tk.Tk()
        app = NeuralNetworkGUI(root)
        root.mainloop()
    else:
        run_cli(args)

if __name__ == "__main__":
    main()
