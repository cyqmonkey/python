import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import os
import numpy as np

batch_size = 100
learning_rate_base = 0.005
learning_rate_decay = 0.99
regularizer = 0.0001
steps = 50000
moving_average_decay = 0.99
model_save_path = "./model/"
model_name = "mnist_model"


def backward(mnist):
    x = tf.placeholder(tf.float32, [
        batch_size,
        mnist_lenet5_forward.image_size,
        mnist_lenet5_forward.image_size,
        mnist_lenet5_forward.num_channels])
    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.output_node])
    y = mnist_lenet5_forward.forward(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        mnist.train.num_examples / batch_size,
        learning_rate_decay,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(steps):
            xs, ys = mnist.train.next_batch(batch_size)
            reshaped_xs = np.reshape(xs, (
                batch_size,
                mnist_lenet5_forward.image_size,
                mnist_lenet5_forward.image_size,
                mnist_lenet5_forward.num_channels))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                print("After {} training step(s), loss on training batch is {}".format(step, loss_value))
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)


if __name__ == '__main__':
     main()
