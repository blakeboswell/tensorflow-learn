import tensorflow as tf
import numpy as np


def model(x_arr, y_arr):

    # declare list of features
    features = [tf.contrib.layers.real_valued_column('x', dimension=1)]

    # declare an estimator
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

    # add np inputs
    input_fn = tf.contrib.learn.io.numpy_input_fn(
            {'x': x_arr}, y_arr, batch_size=4, num_epochs=1000
    )

    # invoke 1000 training steps
    estimator.fit(input_fn=input_fn, steps=1000)

    print(estimator.evaluate(input_fn=input_fn))


if __name__ == '__main__':

    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])

    model(x, y)


