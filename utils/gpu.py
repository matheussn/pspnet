from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import Session
from tensorflow.python.keras.backend import set_session


def set_gpu_limit():
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    set_session(Session(config=config))
