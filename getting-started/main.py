import tensorflow as tf

## example tensors
rank0 = 3                # a rank 0 tensor; this is a scalar with shape []
rank1 = [1., 2., 3.]     # a rank 1 tensor; this is a vecotr with shape [3]
rank2 = [[1., 2., 3.],
         [4., 5., 6.]]   # a rank 2 tensor; a matrix with shape [2, 3]
rank3 = [[[1., 2., 3.]],
         [[7., 8., 9.]]] # a rank 3 tensor with shpape [2, 1, 3]


def simple_graph():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicltly
    print(node1, node2)


def simple_graph_run():
    node1, node2  = tf.constant(3.0), tf.constant(4.0)
    sess = tf.Session()
    print(sess.run([node1, node2]))


def simple_graph_add():
    node1, node2 =  tf.constant(3.0), tf.constant(4.0)
    node3 = tf.add(node1, node2)
    print('node3: {}'.format(node3))
    sess = tf.Session()
    print('sess.run(node3): {}'.format(sess.run(node3)))


def graph_add_ph():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    sess = tf.Session()
    print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))


def graph_add_triple_ph():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    add_and_triple = adder_node * 3
    sess = tf.Session()
    print(sess.run(add_and_triple, {a: 3, b: 4.5}))


def graph_with_vars():
    W = tf.Variable([0.3], tf.float32)
    b = tf.Variable([-0.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W*x + b
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))


def graph_with_vars_loss():
    W = tf.Variable([0.3], tf.float32)
    b = tf.Variable([-0.3], tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    linear_model = W*x + b
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


def graph_with_assign():
    W = tf.Variable([0.3], tf.float32)
    b = tf.Variable([-0.3], tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    linear_model = W*x + b
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
 


if __name__ == '__main__':
    simple_graph()
    simple_graph_run()
    simple_graph_add()
    graph_add_ph()
    graph_add_triple_ph()
    graph_with_vars()
    graph_with_vars_loss()
    graph_with_assign()
