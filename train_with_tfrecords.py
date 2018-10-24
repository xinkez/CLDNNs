# coding=utf-8
import tensorflow as tf
import os
import numpy as np
import cl_layer as Forward
import VGG16
import ResNet


BATCH_SIZE = 32
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
TRAIN_DATA = "/Users/Roy/PycharmProjects/CLDNNs/data.tfrecords"


def backward():
    # 给输入占位

    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE, SEQUENCE_LEN, FRAME_LEN, 1])
    y_ = tf.placeholder(tf.float32,
                        [None, Forward.OUTPUT_NODE])
    y = Forward.forward(x, True, REGULARIZER)
    y = ResNet.resnet_v2_50(y, 18)
    print(y.shape)


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

    print_log = False
    print_process = False
    prtin_test_acc = False

    with tf.Session() as sess:
        train_data, train_label = decode_from_tfrecords(TRAIN_DATA, is_batch=True)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        threads = tf.train.start_queue_runners(sess=sess)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

        ccy = None  # for training data log
        ccy_ = None  # for training data log

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("1")

        if print_process:
            p = open("./process", 'w')
        for i in range(STEPS):
            _data, _label = sess.run([train_data, train_label])
            reshaped_xs = np.reshape(_data, (BATCH_SIZE, SEQUENCE_LEN, FRAME_LEN, 1))
            ys_ = _label
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys_})
            accuracy_score, yy, yy_ = sess.run([accuracy, y, y_], feed_dict={x: reshaped_xs, y_: ys_})


            if i % 1 == 0:
                print("After %d training step(s), loss on training batch is %g, acc is %g."
                      % (step, loss_value, accuracy_score))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

                if prtin_test_acc:
                    a = 1

                if print_process:
                    f.writelines("After %d training step(s), loss on training batch is %g, acc is %g."
                                 % (step, loss_value, accuracy_score))
                    # f.writelines("Test acc is %g \n", test_acc)

            if print_log:
                if ccy is None:
                    ccy = yy
                else:
                    ccy = np.concatenate([ccy, yy], 0)
                if ccy_ is None:
                    ccy_ = yy_
                else:
                    ccy_ = np.concatenate([ccy_, yy_], 0)


            if print_process:
                p.close()
            if print_log:
                f = open('./log_training_data', 'w')
                for k, j in zip(list(ccy), list(ccy_)):
                    f.writelines('y: ' + str(k) + ' ' + str(np.argmax(i)) + '\n')
                    f.writelines('y_: ' + str(j) + ' ' + str(np.argmax(j)) + '\n')
                f.close()


def _parse_function(example_proto):
    features = {"data": tf.FixedLenFeature((), tf.string),
              "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], tf.float32)
    data = np.reshape(data, [SEQUENCE_LEN, FRAME_LEN])
    return data, parsed_features["label"]


def decode_from_tfrecords(filename_queue, is_batch):
    filename_queue = tf.train.string_input_producer([filename_queue])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={"data": tf.FixedLenFeature((), tf.string),
                                                "label": tf.FixedLenFeature((), tf.int64)})
    data = tf.decode_raw(features['data'], tf.float32)
    data = tf.reshape(data, [SEQUENCE_LEN, FRAME_LEN])
    label = tf.cast(features['label'], tf.int64)
    label = tf.one_hot(label, Forward.OUTPUT_NODE, 1, 0)

    if is_batch:
        batch_size = BATCH_SIZE
        min_after_dequeue = 500
        capacity = min_after_dequeue + 3 * batch_size
        data, label = tf.train.shuffle_batch([data, label],
                                              batch_size=batch_size,
                                              num_threads=3,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return data, label


def main():
    backward()

if __name__ == '__main__':
    main()
