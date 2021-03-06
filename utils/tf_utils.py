import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
   x = layers.Conv1D(filters, kernel_size = 1, padding = "valid", name=f"{name}")(x)
   x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
   return layers.Activation("relu",name=f"{name}_relu")(x)

def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)