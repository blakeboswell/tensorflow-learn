import numpy as p
import tensorflow as tf

def model(x_train, y_train):

    # model paramaters
    W = tf.Variable([0.3], tf.float32)
    b = tf.Variable([-0.3], tf.float32)

    # model input and output
    x = tf.placeholder(tf.float32)
    model = W*x + b
    y = tf.placeholder(tf.float32)

    # ssq loss
    loss = tf.reduce_sum(tf.square(model - y)) 

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    curr_W, curr_b, curr_loss = sess.run([W, b, loss],
                                         {x: x_train, y: y_train})
    print('W: {} b: {} loss: {}'.format(curr_W, curr_b, curr_loss))


if __name__ == '__main__':
    
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    model(x_train, y_train)
