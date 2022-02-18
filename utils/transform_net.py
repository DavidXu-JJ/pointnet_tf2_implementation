import tensorflow as tf
import numpy as np
import tf_utils

# transform network
# input_transform_layer
class input_transform_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(input_transform_layer, self).__init__()
        """
        Input XYZ Transform Net, input is B*N*3 gray image
        Return :  Transformation matrix of size 3*K
        """

    def __call__(self, point_cloud, is_training):
        K=3
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value

        # turn B*N*3 to B*N*3*1, expand one dimension, represent channel
        input_image = tf.expand_dims(point_cloud, -1)
        # 64 numbers of [1,3] kernels, stride = 1, with bn and relu
        # input 1 dimensions, output 64 dimensions
        # B*N*3*1 => B*N*1*64
        # maybe this layer can be seen to project the point in 3D dimention onto 64 different lines
        net = tf_utils.conv2d(input_image,64,[1,3],
                             padding="valid",
                             stride = [1,1],
                             bn=True,
                             is_training=is_training)
        # 128 numbers of [1,1] kernels, stride = 1, with bn and relu
        # input 64 dimensions, output 128 dimensions
        # B*N*1*64 => B*N*1*128
        net = tf_utils.conv2d(net,128,[1,1],
                             padding="valid",
                             stride = [1,1],
                             bn=True,
                             is_training=is_training)
        # 1024 numbers of [1,1] kernels, stride = 1, with bn and relu
        # input 128 dimensions, output 1024 dimensions
        # B*N*1*128 => B*N*1*1024
        net = tf_utils.conv2d(net,1024,[1,1],
                             padding="valid",
                             stride = [1,1],
                             bn=True,
                             is_training=is_training)

        # B*N*1*1024 => B*1*1*1024
        net = tf_utils.max_pool2d(net,
                                [num_point,1],
                                padding="valid")
        # B*1*1*1024 => B*1024
        net = tf.reshape(net, [batch_size, -1])
        # input B*1024
        # weight 1024*512
        # bias 512*1
        # output B*512
        net = tf_utils.fully_connected(net, 512,
                                      bn=True,
                                      is_training=is_training,)
        # output B*256
        net = tf_utils.fully_connected(net, 256,
                                      bn=True,
                                      is_training=is_training)

        # add fully connected again without bn and relu
        # output B*9
        weights = tf.Variable(tf.initializers.Constant(0.0)([256, 3 * K],
                                                            dtype=tf.float32))
        biases = tf.Variable(tf.initializers.Constant(0.0)([3 * K],
                                                           dtype=tf.float32))
        # biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten, dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [batch_size, 3, K])

        return transform

#feature_transform_net
class feature_transform_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(feature_transform_layer, self).__init__()
        """ Feature Transform Net, input is BxNx1xK
            Return:
                Transformation matrix of size KxK
        """

    def __call__(self, inputs, is_training):
        K=64
        batch_size = inputs.get_shape()[0].value
        num_point = inputs.get_shape()[1].value
        # input is B*N*1*K

        # 64 numbers of [1,1] kernels, stride = 1, with bn and relu
        # still B*N*1*K
        net = tf_utils.conv2d(inputs, 64, [1,1],
                             padding="valid",
                             stride = [1,1],
                             bn=True,
                             is_training=is_training)
        # 128 numbers of [1,1] kernels, stride = 1, with bn and relu
        # B*N*1*K => B*N*1*128
        net = tf_utils.conv2d(net,128,[1,1],
                             padding="valid",
                             stride = [1,1],
                             bn=True,
                             is_training=is_training)
        # 1024 numbers of [1,1] kernels, stride = 1, with bn and relu
        # B*N*1*128 => B*N*1*1024
        net = tf_utils.conv2d(net,1024,[1,1],
                             padding="valid",
                             stride = [1,1],
                             bn=True,
                             is_training=is_training)

        # B*N*1*1024 => B*1*1*1024
        net = tf_utils.max_pool2d(net,
                                [num_point,1],
                                padding="valid")
        # B*1*1*1024 => B*1024
        net = tf.reshape(net, [batch_size, -1])
        # input B*1024
        # weight 1024*512
        # bias 512*1
        # output B*512
        net = tf_utils.fully_connected(net, 512,
                                      bn=True,
                                      is_training=is_training,)
        # output B*256
        net = tf_utils.fully_connected(net, 256,
                                      bn=True,
                                      is_training=is_training)

        # add fully connected again without bn and relu
        # output B*(64*64)
        weights = tf.Variable(tf.initializers.Constant(0.0)([256, K * K],
                                                            dtype=tf.float32))
        biases = tf.Variable(tf.initializers.Constant(0.0)([K * K],
                                                           dtype=tf.float32))
        # biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten, dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [batch_size, K, K])

        return transform
