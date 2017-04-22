import tensorflow as tf
import numpy as np


def custom_model(features, labels, mode):

    W = tf.get_variable('W', [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    y = W*features['x'] + b

    # loss sub-graph
    loss  = tf.reduce_sum(tf.square(y - labels))

    # train sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                     tf.assign_add(global_step, 1))

    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality
    return tf.contrib.learn.ModelFnOps(
            mode=mode,
            predictions=y,
            loss=loss,
            train_op=train
    )


if __name__ == '__main__':

    estimator = tf.contrib.learn.Estimator(model_fn=custom_model)

    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])

    input_fn = tf.contrib.learn.io.numpy_input_fn(
            {'x': x}, y, 4, num_epochs=1000
    )
    
    # train
    estimator.fit(input_fn=input_fn, steps=1000)

    # evaluate
    print(estimator.evaluate(input_fn=input_fn, steps=10))

