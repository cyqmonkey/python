import tensorflow as tf

input_node = 784
output_node = 10
layer_node = 500


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    w1 = get_weight([input_node, layer_node], regularizer)
    b1 = get_bias([layer_node])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([layer_node, output_node], regularizer)
    b2 = get_bias([output_node])
    y = tf.matmul(y1, w2) + b2
    return y
