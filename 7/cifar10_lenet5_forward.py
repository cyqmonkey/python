import tensorflow as tf
image_size = 24
num_channels = 3
conv1_size = 5
conv1_kernel_num = 64
conv2_size = 5
conv2_kernel_num = 64
fc1_size = 384
fc2_size = 192
output_node = 10


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.1, shape=shape))
    return b


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


def forward(x, train, regularizer):
    conv1_w = get_weight([conv1_size, conv1_size, num_channels, conv1_kernel_num], regularizer)
    conv1_b = get_bias([conv1_kernel_num])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.bias_add(conv1, conv1_b), training=train))
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight([conv2_size, conv2_size, conv1_kernel_num, conv2_kernel_num], regularizer)
    conv2_b = get_bias([conv2_kernel_num])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.bias_add(conv2, conv2_b), training=train))
    pool2 = max_pool_2x2(relu2)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, fc1_size], regularizer)
    fc1_b = get_bias([fc1_size])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([fc1_size, fc2_size], regularizer)
    fc2_b = get_bias([fc2_size])
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
    if train:
        fc2 = tf.nn.dropout(fc2, 0.5)

    fc_w = get_weight([fc2_size, output_node], regularizer)
    fc_b = get_bias([output_node])
    y = tf.matmul(fc2, fc_w) + fc_b
    return y

