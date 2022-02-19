import tensorflow as tf
import numpy as np
import tf_utils
from transform_net import input_transform_layer, feature_transform_layer


class pointnet_seg(tf.keras.Model):
    # Segmentation PointNet, input is BxNx3, output BxN*50
    def __init__(self):
        super(pointnet_seg).__init__()
        self.input_transform = input_transform_layer()
        self.feature_transform = feature_transform_layer()

    def __call__(self, point_cloud, is_training):
        # input is B*N*3
        # output B*N*50
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[0].value
        end_points = {}

        # get a transformation matrix
        input_transform = self.input_transform(point_cloud, is_training)
        # still B*N*3
        point_cloud_transformed = tf.matmul(point_cloud, input_transform)
        # B*N*3 => B*N*3*1
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        # B*N*3*1 => B*N*1*64
        net = tf_utils.conv2d(input_image, 64, [1, 3],
                              padding="valid",
                              stride=[1, 1],
                              bn=True,
                              is_training=is_training)

        # still B*N*1*64
        net = tf_utils.conv2d(net, 64, [1, 1],
                              padding="valid",
                              stride=[1, 1],
                              bn=True,
                              is_training=is_training)

        # get a transformation matrix
        feature_transform = self.feature_transform(net, is_training)

        end_points['transform'] = feature_transform

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
                              bn=True,
                              is_training=is_training)

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
                            bn = True, is_training = is_training)
        # B*N*1*512 => B*N*1*256
        net = tf_utils.conv2d(net, 256, [1, 1],
                              padding="valid", stride = [1,1],
                              bn = True, is_training = is_training)
        # B*N*1*256 => B*N*1*128
        net = tf_utils.conv2d(net, 128, [1, 1],
                              padding="valid", stride = [1,1],
                              bn = True, is_training = is_training)
        # still B*N*1*128
        net = tf_utils.conv2d(net, 128, [1, 1],
                              padding="valid", stride = [1,1],
                              bn = True, is_training = is_training)

        net = tf_utils.conv2d(net, 50, [1, 1],
                              padding="valid",stride = [1, 1],
                              activation_fn = None)

        return net, end_points
