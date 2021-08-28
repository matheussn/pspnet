import tensorflow

print(f"Tensorflow: {tensorflow.__version__}")
print(f"Keras: {tensorflow.keras.__version__}")

print(f"Devices: {tensorflow.config.list_physical_devices()}")
