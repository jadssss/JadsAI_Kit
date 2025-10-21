import argparse
import sys
from core.model_builder import ModelBuilder
from core.data_loader import DataLoader
from core.trainer import Trainer
from core.gpu_checker import GPUChecker
from core.predictor import Predictor
from utils.logger import get_logger

def run_cli(args):
    """CLI интерфейс"""
    logger = get_logger(__name__)
    model_builder = ModelBuilder(verbose=args.verbose)
    data_loader = DataLoader()
    trainer = Trainer(model_builder)
    gpu_checker = GPUChecker(verbose=args.verbose)
    predictor = Predictor(model_builder, data_loader)
    
    try:
        if args.test_gpu:
            gpu_checker.test_gpu()
            return

        if args.captcha_dir:
            data = data_loader.load_captcha_dataset(args.captcha_dir)
        elif args.load_data:
            data = data_loader.load_data(custom_path=args.load_data)
        else:
            data = data_loader.load_data(dataset='mnist')

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
            'conv_layers': conv_layers if conv_layers else [
                {'filters': 32, 'kernel': 3, 'stride': 1, 'padding': 'same'},
                {'filters': 64, 'kernel': 3, 'stride': 1, 'padding': 'same'},
                {'filters': 128, 'kernel': 3, 'stride': 1, 'padding': 'same'}
            ],
            'dense_layers': [int(x) for x in args.dense_layers.split(',') if x] if args.dense_layers else [128],
            'dropout': args.dropout,
            'batch_norm': args.batch_norm,
            'activation': args.activation,
            'optimizer': args.optimizer,
            'lr': args.lr,
            'loss': args.loss,
            'metrics': ['accuracy'] if args.loss != 'ctc_loss' else []
        }

        model_builder.create_model(args.architecture, config, input_shape=(32, 128, 1), num_classes=70)

        if args.train:
            trainer.train_model(data[0], epochs=args.epochs, batch_size=args.batch_size, validation_data=data[1])
        
        if args.evaluate:
            loss, _ = trainer.evaluate_model(data[1])
            logger.info(f"Оценка: loss={loss:.4f}")
        
        if args.save:
            model_builder.save_model(args.save)
        
        if args.load:
            model_builder.load_model(args.load)

        if args.predict:
            prediction = predictor.predict(args.predict)
            logger.info(f"Предсказание для {args.predict}: {prediction}")

    except Exception as e:
        logger.error(f"Ошибка в CLI: {e}")
        sys.exit(1)
