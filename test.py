import tensorflow as tf

print(f"Tensorflow: {tf.__version__}")
print(f"Keras: {tf.keras.__version__}")

print(f"Devices: {tf.config.list_physical_devices()}")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
