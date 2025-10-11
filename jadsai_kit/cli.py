# jadsai_kit/cli.py
from .model_builder import create_cnn_model
from .gpu_utils import check_gpu

def cli_mode(args):
    print("JadsAI_Kit CLI Mode")
    if args.help:
        print("""
        JadsAI_Kit CLI Usage:
        --create-cnn : Create a CNN model
        --check-gpu : Check GPU availability
        Example: python -m jadsai_kit cli --create-cnn
        """)
        return
    
    if args.create_cnn:
        print("Creating CNN model...")
        model = create_cnn_model()
        model.summary()
    
    if args.check_gpu:
        check_gpu()
