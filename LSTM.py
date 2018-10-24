import tensorflow as tf
import cl_layer
import train_with_tfrecords


batch_size = train_with_tfrecords.BATCH_SIZE
n_input = 16
n_step = 10
n_hidden_units = 128
n_class = cl_layer.OUTPUT_NODE
KEEP_PROB = 0.5
# weights ans biases
weight = {
    'in': tf.Variable(tf.random_normal([n_input, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_class]))
}
bias = {
    'in': tf.constant(0.1, shape=[n_hidden_units,]),
    'out': tf.constant(0.1, shape=[n_class,])
}


# RNN
def rnn(X, weight, biases, train):
    X = tf.reshape(X, [-1, n_input])
    X_in = tf.matmul(X, weight['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_step, n_hidden_units])
    stack_rnn = []
    for i in range(2):
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        if train:
            cell = rnn.DropoutWrapper(cell, output_keep_prob=KEEP_PROB)
        stack_rnn.append(cell)

    lstm_cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(states[1], weight['out']) + biases['out']
    return results