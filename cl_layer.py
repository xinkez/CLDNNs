# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import train_with_tfrecords as tr

# 时间卷积参数
tCONV_KENEL_NUM = 39
tCONV_KENEL_SIZE = 400
tCONV_STRIDE = 1
tCONV_POOLING_SIZE = 81
tCONV_POOLING_STRIDE = 1
NUM_CHANNELS = 1

# 频率卷积参数
fCONV_KENEL_NUM = 256
fCONV_KENEL_SIZE = 8
fCONV_STRIDE = 1
fCONV_POOLING_SIZE = 3
fCONV_POOLING_STRIDE = 1

# LTSM层参数
LSTM_NUM_UNITS = 280
KEEP_PROB = 0.5
NUM_LAYERS = 2
BATCH_SIZE = 32
NUM_STEPS = 1

# 分类个数
OUTPUT_NODE = 18


def forward(x, train, regularizer):

    # 时间卷积层
    conv1_w = get_weight([1, tCONV_KENEL_SIZE, 1, tCONV_KENEL_NUM], regularizer)
    conv1_b = get_bias(tCONV_KENEL_NUM)
    conv1 = conv_layer(x, conv1_w, conv1_b, 2, name="conv1")
    pool1_shape = [1, 1, tCONV_POOLING_SIZE, 1]
    pool1 = max_pool_2x2(conv1, shape=pool1_shape, step=tCONV_POOLING_STRIDE ,name="pool1")

    pool1 = tf.reshape(pool1, [32, 27, 39, 1])

    # 频率卷积层
    conv2_1_w = get_weight([1, fCONV_KENEL_SIZE, tCONV_KENEL_NUM, fCONV_KENEL_NUM], regularizer)
    conv2_1_b = get_bias(fCONV_KENEL_NUM)
    conv2_1 = conv_layer(pool1, conv2_1_w, conv2_1_b, 1, name="conv2_1")
    pool2_shape = [1, 1, fCONV_POOLING_SIZE, 1]
    pool2 = max_pool_2x2(conv2_1, shape=pool2_shape, step=fCONV_POOLING_STRIDE, name="pool2")


    return pool2


def conv_layer(x, w, b, step, name):
    with tf.variable_scope(name):
        conv = tf.nn.conv2d(x, w, [1, 1, step, 1], padding='SAME')
        a_mean, a_var = tf.nn.moments(conv, axes=[0,1],keep_dims=True)
        conv_bn = tf.nn.batch_normalization(conv ,a_mean,a_var,offset=None,scale=1,variance_epsilon=0)
        result = tf.nn.relu(tf.nn.bias_add(conv_bn, b))
        return result


def fconv_layer(x, w, b, step, name):
    with tf.variable_scope(name):
        conv = tf.nn.conv2d(x, w, [1, 1, step, 1], padding='SAME')
        a_mean, a_var = tf.nn.moments(conv, axes=[0,1],keep_dims=True)
        conv_bn = tf.nn.batch_normalization(conv ,a_mean,a_var,offset=None,scale=1,variance_epsilon=0)
        result = tf.nn.relu(tf.nn.bias_add(conv_bn, b))
        return result


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def avg_pool(x, name):
    with tf.variable_scope(name):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x, shape, step, name):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=shape, strides=[1, step, step, 1], padding='SAME')




