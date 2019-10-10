import tensorflow as tf
import cifar10_input
import cifar10_lenet5_forward
import os

batch_size = 100
learning_rate_base = 0.01
learning_rate_decay = 0.9
regularizer = 0.004
steps = 5000
moving_average_decay = 0.99
model_save_path = "./model/"
model_name = "cifar10_model"
num_examples = 50000


def backward():
    x = tf.placeholder(tf.float32, [
        batch_size,
        cifar10_lenet5_forward.image_size,
        cifar10_lenet5_forward.image_size,
        cifar10_lenet5_forward.num_channels])
    y_ = tf.placeholder(tf.int32, [batch_size])
    y = cifar10_lenet5_forward.forward(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        num_examples / batch_size,
        learning_rate_decay,
        staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
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

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        xs, ys = cifar10_input.distorted_inputs(batch_size)
        for i in range(steps):
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs.eval(), y_: ys.eval()})
            if i % 100 == 0:
                print("After {} training step(s), loss on training batch is {}".format(step, loss_value))
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)
        coord.request_stop()
        coord.join(threads)

def main():
    backward()


if __name__ == '__main__':
     main()
