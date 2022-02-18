import numpy as np
import tensorflow as tf

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.Variable(initializer(shape), dtype=dtype, name = name)
  return var

def _variable_with_weight_decay(name, shape, stddev, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      use_xavier: bool, whether to use xavier initializer
    Returns:
      Variable Tensor
    """
    if use_xavier:
        initializer = tf.initializers.GlorotUniform()
    else:
        initializer = tf.initializers.TruncatedNormal(stddev= stddev)
    var = _variable_on_cpu(name, shape, initializer)
    return var

def conv2d( inputs,
            num_output_channels,
            kernel_size,
            stride = [1, 1],
            padding = "SAME",
            use_xavier = True,
            stddev = 1e-3,
            activation_fn = tf.nn.relu,
            bn = False,
            is_training = None ):

    """ 2D convolution with non-linear operation.
    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      activation_fn: function
      bn: bool, whether to use batch norm
      is_training: bool Tensor variable
    Returns:
      Variable tensor
    """

    kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay( "weights",
                                          shape=kernel_shape,
                                          use_xavier=use_xavier,
                                          stddev=stddev)
    stride_h, stride_w = stride
    outputs = tf.nn.conv2d(inputs,kernel,[1, stride_h, stride_w, 1],padding = padding)
    biases = _variable_on_cpu('biases',[num_output_channels],tf.initializers.Constant(0.0))

    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs,is_training = is_training)

    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs

def max_pool2d(inputs,
               kernel_size,
               stride=[2,2],
               padding="valid"):
    """ 2D max pooling.
    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """

    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize = [1,kernel_h,kernel_w,1],
                             strides = [1,stride_h,stride_w,1],
                             padding=padding)
    return outputs

def fully_connected(inputs,
                    num_outputs,
                    use_xavier,
                    stddev=1e-3,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay("weights",
                                          shape=[num_input_units,num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev)
    outputs = tf.matmul(inputs,weights)
    biases = _variable_on_cpu("biases",[num_outputs],
                              tf.initializers.Constant(0.0))
    #last dimention is output filters
    outputs = tf.nn.bias_add(outputs,biases)

    if bn:
        outputs = tf.keras.layers.BatchNormalization(outputs, is_training=is_training)

    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs


def dropout(inputs,
            is_training,
            keep_prob = 0.5,
            noise_shape = None):
    """ Dropout layer.
    Args:
      inputs: tensor
      is_training: boolean tf.Variable
      keep_prob: float in [0,1]
      noise_shape: list of ints
    Returns:
      tensor variable
    """
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs