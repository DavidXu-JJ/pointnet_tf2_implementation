import tensorflow as tf
import numpy as np
import math
import json
import sys
import os
from tqdm import tqdm
from glob import glob

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, './utils'))

import tensorflow as tf
import numpy as np
import tf_utils
from transform_net import input_transform_layer, feature_transform_layer

# Segmentation PointNet, input is BxNx3, output BxN*50

def get_pointnet_seg(num_point, num_classes):
    # input is B*N*3
    # output B*N*50
    input_points = tf.keras.Input(shape=(None, 3))


    # get a transformation matrix
    input_transform = input_transform_layer()(input_points)
    # still B*N*3
    point_cloud_transformed = tf.matmul(input_points, input_transform)
    # B*N*3 => B*N*3*1
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    # B*N*3*1 => B*N*1*64
    net = tf_utils.conv2d(input_image, 64, [1, 3],
                          padding="valid",
                          stride=[1, 1],
                          bn=True)

    # still B*N*1*64
    net = tf_utils.conv2d(net, 64, [1, 1],
                          padding="valid",
                          stride=[1, 1],
                          bn=True)

    # get a transformation matrix
    feature_transform = feature_transform_layer()(net)

    # B*N*1*64 => B*N*64
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), feature_transform)

    # B*N*64 => B*N*1*64
    point_feat = tf.expand_dims(net_transformed, [2])

    # these multi-layer perceptrons maybe strive to find the important point
    # or important feature in high dimension
    # still B*N*1*64
    net = tf_utils.conv2d(point_feat, 64, [1, 1],
                          padding="valid",
                          stride=[1, 1],
                          bn=True)

    # B*N*1*64 => B*N*1*128
    net = tf_utils.conv2d(net, 128, [1, 1],
                          padding="valid",
                          stride=[1, 1],
                          bn=True)

    # B*N*1*128 => B*N*1*1024
    net = tf_utils.conv2d(net, 1024, [1, 1],
                          padding="valid",
                          stride=[1, 1],
                          bn=True)

    # extract global feature
    # B*N*1*1024 => B*1*1*1024
    global_feat = tf_utils.max_pool2d(net, [num_point, 1],
                                      padding="valid")
    # B*1*1*1024 => B*N*1*1024
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])

    # B*N*1*64 and B*N*1*1024 => B*N*1*1088
    concat_feat = tf.concat([point_feat, global_feat_expand],3)

    # B*N*1*1088 => B*N*1*512
    net = tf_utils.conv2d(concat_feat, 512, [1, 1],
                          padding="valid", stride = [1,1],
                          bn = True)
    # B*N*1*512 => B*N*1*256
    net = tf_utils.conv2d(net, 256, [1, 1],
                          padding="valid", stride = [1,1],
                          bn = True)
    # B*N*1*256 => B*N*1*128
    net = tf_utils.conv2d(net, 128, [1, 1],
                          padding="valid", stride = [1,1],
                          bn = True)
    # still B*N*1*128
    net = tf_utils.conv2d(net, 128, [1, 1],
                          padding="valid", stride = [1,1],
                          bn = True)

    # B*N*1*128 => B*N*1*50
    # question: why this layer don't need training?
    net = tf_utils.conv2d(net, num_classes, [1, 1],
                          padding="valid",stride = [1, 1],
                          activation_fn = None)

    # B*N*1*50 => B*N*50
    net = tf.squeeze(net, [2])

    return tf.keras.Model(input_points, net)

a=get_pointnet_seg(3,4)