import tensorflow as tf
import numpy as np
import tf_utils
from transform_net import input_transform_layer, feature_transform_layer

class pointnet_cls(tf.keras.Model):
    def __init__(self):
        super(pointnet_cls).__init__()
        self.input_transform = input_transform_layer()
        self.feature_transform = feature_transform_layer()

    def __call__(self, point_cloud, is_training):
        # input is B*N*3
        # output B*40
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[0].value
        end_points= {}

        #get a transformation matrix
        input_transform = self.input_transform(point_cloud, is_training)
        # still B*N*3
        point_cloud_transformed = tf.matmul(point_cloud, input_transform)
        # B*N*3 => B*N*3*1
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        # B*N*3*1 => B*N*1*64
        net = tf_utils.conv2d(input_image, 64, [1,3],
                              padding = "valid",
                              stride = [1,1],
                              bn = True,
                              is_training = is_training)

        # still B*N*1*64
        net = tf_utils.conv2d(net, 64, [1,1],
                              padding = "valid",
                              stride = [1,1],
                              bn = True,
                              is_training = is_training)

        #get a transformation matrix
        feature_transform = self.feature_transform(net, is_training)

        end_points['transform'] = feature_transform

        # B*N*1*64 => B*N*64
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), feature_transform)

        # B*N*64 => B*N*1*64
        net_transformed = tf.expand_dims(net_transformed, [2])

        # these multi-layer perceptrons maybe strive to find the important point
        # or important feature in high dimension
        # still B*N*1*64
        net = tf_utils.conv2d(net_transformed, 64, [1, 1],
                              padding = "valid",
                              stride = [1, 1],
                              bn = True,
                              is_training = is_training)

        # B*N*1*64 => B*N*1*128
        net = tf_utils.conv2d(net, 128, [1, 1],
                              padding="valid",
                              stride=[1, 1],
                              bn=True,
                              is_training=is_training)

        # B*N*1*128 => B*N*1*1024
        net = tf_utils.conv2d(net, 1024, [1, 1],
                              padding="valid",
                              stride=[1, 1],
                              bn=True,
                              is_training=is_training)

        # Symmetric function: max pooling
        # try to make model invariant to permutation of point cloud
        # B*N*1*1024 => B*1*1024
        net = tf_utils.max_pool2d(net, [num_point, 1],
                                  padding = "valid")
        # B*1*1024 => B*1024
        net = tf.reshape(net, [batch_size, -1])

        # the fully-connect layers maybe extract information from
        # redundant high dimension output
        # input B*1024
        # weight = 1024*512
        # output B*512
        net = tf_utils.fully_connected(net, 512,
                                       bn = True,
                                       is_training = is_training)

        net = tf_utils.dropout(net, keep_prob=0.7, is_training=is_training)

        # input B*512
        # weight = 512*256
        # output B*256
        net = tf_utils.fully_connected(net, 256,
                                       bn = True,
                                       is_training = is_training)
        
        net = tf_utils.dropout(net, keep_prob=0.7, is_training=is_training)

        # input B*256
        # weight = 256*40
        # output B*40
        # no bn, relu
        net = tf_utils.fully_connected(net, 40, activation_fn=None)

        return net, end_points


