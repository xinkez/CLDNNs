# coding:utf-8
import time
import tensorflow as tf
import train as tr
import numpy as np
import Forward
import DataManage as dt

TEST_INTERVAL_SECS = 5
BATCH_SIZE = tr.BATCH_SIZE
SEQUENCE_LEN = tr.SEQUENCE_LEN
FRAME_LEN = tr.FRAME_LEN

TEST_DATA_PATH = ""

TEST_BATCH = 10

def main():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,
                           [TEST_BATCH, SEQUENCE_LEN, FRAME_LEN, 1])
        y_ = tf.placeholder(tf.float32,
                            [None, Forward.OUTPUT_NODE])
        y = Forward.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(tr.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(tr.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    file_name = dt.shuffle(TEST_DATA_PATH)
                    epoch = 10
                    for batch_count in range(epoch):
                        xs_, ys_ = dt.next_batch(file_name, BATCH_SIZE, batch_count)
                        reshaped_x = np.reshape(xs_, (
                            [BATCH_SIZE, SEQUENCE_LEN, FRAME_LEN, 1]))
                        accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_x, y_: ys_})
                        print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


if __name__ == '__main__':
    main()
