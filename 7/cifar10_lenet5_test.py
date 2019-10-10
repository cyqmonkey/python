import tensorflow as tf
import cifar10_input
import cifar10_lenet5_forward
import cifar10_lenet5_backward
import numpy as np


def test():
    with tf.Graph().as_default() as g:
        test_num_examples =  cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        x = tf.placeholder(tf.float32, [
            test_num_examples,
            cifar10_lenet5_forward.image_size,
            cifar10_lenet5_forward.image_size,
            cifar10_lenet5_forward.num_channels])
        y_ = tf.placeholder(tf.int32, [test_num_examples])
        y = cifar10_lenet5_forward.forward(x, False, cifar10_lenet5_backward.regularizer)

        ema = tf.train.ExponentialMovingAverage(cifar10_lenet5_backward.moving_average_decay)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        top_k_op = tf.nn.in_top_k(y, y_, 1)
        # num = test_num_examples / cifar10_lenet5_backward.batch_size
        # true_count = 0
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(cifar10_lenet5_backward.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                xs, ys = cifar10_input.inputs('test', test_num_examples)
                predictions = sess.run(top_k_op, feed_dict={x: xs.eval(), y_: ys.eval()})
                precision = np.sum(predictions) / test_num_examples
                print("After {} training step(s), test accuracy = {}".format(global_step, precision))
            else:
                print("No checkpoint file found")
                return


def main():
    test()


if __name__ == '__main__':
    main()