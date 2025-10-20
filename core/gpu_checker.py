import tensorflow as tf
from utils.logger import get_logger

class GPUChecker:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.logger = get_logger(__name__)
        self.gpu_available = self.check_gpu()

    def check_gpu(self):
        """Проверка GPU и CUDA"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                if self.verbose:
                    self.logger.info(f"Обнаружены GPU: {[gpu.name for gpu in gpus]}")
                    self.logger.info(f"Версия TensorFlow: {tf.__version__}")
                    cuda_version = tf.sysconfig.get_build_info().get('cuda_version', 'Unknown')
                    cudnn_version = tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown')
                    self.logger.info(f"CUDA: {cuda_version}, cuDNN: {cudnn_version}")
                return True
            else:
                self.logger.warning("GPU не обнаружены. Проверьте:")
                self.logger.warning("1. CUDA 12.4: ls /usr/local/cuda-12.4/lib64/libcudart*")
                self.logger.warning("2. cuDNN 9.0+: ls /usr/local/cuda-12.4/lib64/libcudnn*")
                self.logger.warning("3. LD_LIBRARY_PATH: echo $LD_LIBRARY_PATH")
                self.logger.warning("4. TensorFlow: tensorflow[and-cuda]==2.17.0")
                self.logger.warning("5. Для Debian 13: установите libtinfo5 из http://deb.debian.org/debian/pool/main/n/ncurses/libtinfo5_6.4-4_amd64.deb")
                self.logger.warning("См. https://www.tensorflow.org/install/gpu")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка проверки GPU: {e}")
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
                self.logger.info(f"Тест GPU: матричный расчёт 10000x10000 завершён за {elapsed:.3f} сек")
        except Exception as e:
            self.logger.error(f"Ошибка теста GPU: {e}")
            raise
