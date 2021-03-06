import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

MAX_STEPS = 1000
LEARN_RATE = 0.001
DROP_OUT = 0.9
DATA_DIR = '../data/MNIST_data'
LOG_DIR = '../log/mnist_summary'


def train():

    mnist = read_data_sets(DATA_DIR, one_hot=True, fake_data=False)
    sess = tf.Session()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    def weight_variable(shape):
        ''' create a weight var with appropriate initialization
        '''
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        ''' create a bias variable with appropriate initialization
        '''
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        ''' attach summaries to Tensor
        '''
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stdev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('min', tf.reduce_max(var))
            tf.summary.scalar('max', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name,
                  act=tf.nn.relu):
        ''' reusable code for making a simple neural net layer
        '''
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations 

    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(
                cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test')
    init = tf.global_variables_initializer()
    sess.run(init)

    def feed_dict(train):
        if train:
            xs, ys = mnist.train.next_batch(100, fake_data=False)
            k = DROP_OUT
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(MAX_STEPS):
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step {}: {}'.format(i, acc))
        else:
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run(
                        [merged, train_step],
                        feed_dict=feed_dict(True),
                        options=run_options,
                        run_metadata=run_metadata
                )
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                summary, _ = sess.run(
                        [merged, train_step],
                        feed_dict=feed_dict(True)
                )
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    train()


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)

