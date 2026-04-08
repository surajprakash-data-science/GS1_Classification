import tensorflow as tf
print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))