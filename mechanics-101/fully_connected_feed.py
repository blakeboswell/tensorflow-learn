import math
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE ** 2


def inference(images, hidden1_units, hidden2_units):
    ''' build mnist model with inference
        
        Args:
            images: Images placeholder
            hidden1_units: size of first hidden layer
            hiddne2_units: size of second layer

        Returns:
            softmax_linear: Output tensor with the computed logits
    '''
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
                tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                    stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                name='weights')
        baises = tf.Variable(tf.zeros([hidden_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units],
                                    stddev=1.0 / math.sqrt(float(hidden1_units))),
                name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
                tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                    stddev=1.0 / math.sqrt(float(hidden2_units))),
                name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                           name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    return logits


def loss(logits, labels):
    '''
    '''
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


