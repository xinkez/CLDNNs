import tensorflow as tf
import numpy as np
import cl_layer as Forward


def _parse_function(example_proto):
    features = {"data": tf.FixedLenFeature((), tf.string),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], tf.float32)
    return data, parsed_features["label"]

def decode_from_tfrecords(filename_queue, is_batch):
    filename_queue = tf.train.string_input_producer([filename_queue])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={"data": tf.FixedLenFeature((), tf.string),
                                                "label": tf.FixedLenFeature((), tf.int64)})
    data = tf.decode_raw(features['data'], tf.float32)
    data = tf.reshape(data, [27, 560])
    label = tf.cast(features['label'], tf.int64)
    label = tf.one_hot(label, 18, 1, 0)

    if is_batch:
        batch_size = 32
        min_after_dequeue = 500
        capacity = min_after_dequeue + 3 * batch_size
        data, label = tf.train.shuffle_batch([data, label],
                                              batch_size=batch_size,
                                              num_threads=3,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return data, label

def load_tfrecords(srcfile):
    sess = tf.Session()
    '''
    dataset = tf.data.TFRecordDataset(srcfile)  # load tfrecord file
    dataset = dataset.map(_parse_function)  # parse data into tensor
    dataset = dataset.repeat(1)  # repeat for 2 epoches
    dataset = dataset.batch(3)  # set batch_size = 5
    '''

    #iterator = dataset.make_one_shot_iterator()
    #next_data = iterator.get_next()
    train_data, train_label = decode_from_tfrecords(srcfile, True)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(10):
            data = sess.run(train_data)
            label = sess.run(train_label)
            print(data.shape)
            print(label)




if __name__  == "__main__":
    load_tfrecords(srcfile="./data.tfrecords")
