import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .tf_utils import conv_block, mlp_block
from .transform_net import transformation_block


def get_shape_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(None, 3))

    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )

    segmentation_features = conv_block(
        segmentation_input , filters = 128, name = "segmentation_features"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1,activation ="softmax", name ="segmentation_head"
    )(segmentation_features)

    return keras.Model(input_points, outputs)