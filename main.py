import argparse
import tensorflow as tf
from core.data_loader import DataLoader
from core.model_builder import ModelBuilder
from core.trainer import Trainer
from utils.logger import get_logger

def main():
    parser = argparse.ArgumentParser(description="Капча-распознаватель CLI")
    parser.add_argument('--mode', choices=['cli', 'gui'], default='cli')
    parser.add_argument('--architecture', choices=['crnn'], default='crnn')
    parser.add_argument('--captcha-dir', type=str, help='Путь к папке с капчами')
    parser.add_argument('--train', action='store_true', help='Обучить модель')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--conv-layers', type=str, default='32,3,1,same;64,3,1,same;128,3,1,same')
    parser.add_argument('--dense-layers', type=str, default='128')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch-norm', action='store_true')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss', type=str, default='ctc_loss')
    parser.add_argument('--save', type=str, help='Путь для сохранения модели')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logger = get_logger(__name__)

    if args.mode == 'cli':
        try:
            # Проверка GPU
            gpus = tf.config.list_physical_devices('GPU')
            logger.info(f"Обнаружены GPU: {gpus}")
            logger.info(f"Версия TensorFlow: {tf.__version__}")

            # Загрузка данных
            data_loader = DataLoader()
            train_data, test_data = data_loader.load_captcha_dataset(args.captcha_dir, img_size=(128, 32), grayscale=True)

            # Создание модели
            model_config = {
                'conv_layers': [{'filters': int(l.split(',')[0]), 'kernel': int(l.split(',')[1]), 
                                 'stride': int(l.split(',')[2]), 'padding': l.split(',')[3]} 
                                for l in args.conv_layers.split(';')],
                'dense_layers': [int(x) for x in args.dense_layers.split(',')],
                'dropout': args.dropout,
                'batch_norm': args.batch_norm,
                'activation': args.activation,
                'optimizer': args.optimizer,
                'lr': args.lr,
                'loss': args.loss,
                'metrics': []
            }
            model_builder = ModelBuilder(model_config, len(data_loader.alphabet))
            model = model_builder.build()

            # Обучение
            if args.train:
                trainer = Trainer(model, loss=args.loss, optimizer=args.optimizer, lr=args.lr)
                trainer.train(train_data, test_data, args.epochs, args.batch_size)

            # Сохранение модели
            if args.save:
                model.save(args.save)
                logger.info(f"Модель сохранена в {args.save}")
        except Exception as e:
            logger.error(f"Ошибка в CLI: {e}")
            raise

if __name__ == '__main__':
    main()
