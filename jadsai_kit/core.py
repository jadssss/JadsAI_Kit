#jadsai_kit/core.py
import argparse
import sys
from .gui import JadsAIKitGUI
from .cli import cli_mode
from .gpu_utils import check_gpu

def main():
    parser = argparse.ArgumentParser(description="JadsAI_Kit: Neural Network Builder")
    parser.add_argument('mode', choices=['gui', 'cli'], help="Run in GUI or CLI mode")
    parser.add_argument('--create-cnn', action='store_true', help="Create a CNN model (CLI)")
    parser.add_argument('--check-gpu', action='store_true', help="Check GPU availability (CLI)")
    parser.add_argument('--help', action='store_true', help="Show help message (CLI)")
    
    args = parser.parse_args()
    
    # Проверка GPU при запуске
    check_gpu()
    
    if args.mode == 'gui':
        JadsAIKitGUI().run()
    elif args.mode == 'cli':
        cli_mode(args)

if __name__ == "__main__":
    main()
