# coding=utf-8
import tensorflow as tf
import os
import numpy as np
import Forward
import DataManage as dt


BATCH_SIZE = 10
# 声音预处理参数
TOTAL_NUM = 1116
FRAME_LEN = 560
FRAME_MOV = 160
SEQUENCE_LEN = 27

LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 100
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "/Users/Roy/PycharmProjects/CLDNNs/model"
MODEL_NAME = "cldnn_model"

DATA_TRAIN_PATH = "/Users/Roy/PycharmProjects/CLDNN/audio/trainer.list"


def backward():
    # 给输入占位
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE, SEQUENCE_LEN, FRAME_LEN, 1])
    y_ = tf.placeholder(tf.float32,
                        [None, Forward.OUTPUT_NODE])
    y = Forward.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # 定义loss function
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 定义learning rate
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        int(TOTAL_NUM / BATCH_SIZE),
        LEARNING_RATE_DECAY,
        staircase=True)
    # 定义优化器
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    # 定义正确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:

            saver.restore(sess, ckpt.model_checkpoint_path)
            file_name = dt.shuffle(DATA_TRAIN_PATH)
            epoch = int(1116/BATCH_SIZE)
            for i in range(STEPS):
                for batch_count in range(epoch):
                    xs_, ys_ = dt.next_batch(file_name, BATCH_SIZE, batch_count)
                    reshaped_xs = np.reshape(xs_, (BATCH_SIZE, SEQUENCE_LEN, FRAME_LEN, 1))
                    _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys_})
                    accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_xs, y_: ys_})
                    if i % 1 == 0:
                        print("After %d training step(s), loss on training batch %g is %g, acc is %g." % (step, batch_count, loss_value, accuracy_score))
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    backward()

if __name__ == '__main__':
    main()
