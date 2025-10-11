# jadsai_kit/gpu_utils.py
import tensorflow as tf
import platform

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    os_type = platform.system()
    if os_type not in ['Linux', 'Windows']:
        print(f"Warning: {os_type} is not officially supported. Functionality may be limited.")
    if gpus:
        print(f"GPU detected: {gpus}")
        return True
    else:
        print("No GPU detected, using CPU.")
        return False
