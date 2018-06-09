import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
testing_size = 256
img_size = 28
num_classes = 10

X = tf.placeholder("float", [None, img_size,img_size,1])
Y = tf.placeholder("float", [None, num_classes])
mnist = input_data.read_data_sets("data", one_hot=True)
x_train,x_test,y_train,y_test = mnist.train.images,mnist.test.images,mnist.train.labels,mnist.test.labels

x_train = x_train.reshape(-1,img_size, img_size, 1)
x_test = x_test.reshape(-1,img_size, img_size, 1)

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

w = init_weight([3, 3, 1, 32])
w2 = init_weight([3, 3, 32, 64])
w3 = init_weight([3, 3, 64, 128])
w4 = init_weight([128*4*4, 625])
w_o = init_weight([625, num_classes])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):

    conv1 = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')
    conv1_a = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, p_keep_conv)

    conv2 = tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding='SAME')
    conv2_a = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, p_keep_conv)

    conv3=tf.nn.conv2d(conv2, w3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    FC_layer = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    FC_layer = tf.reshape(FC_layer, [-1, w4.get_shape().as_list()[0]])
    FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)


    output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
    output_layer = tf.nn.dropout(output_layer, p_keep_hidden)

    result = tf.matmul(output_layer, w_o)
    return result

py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        # this takes in values an returns a tuple list_a  = [1,2,3] and list_b = [4,5,6]
        # zipped=zip(list_a,list_b) == [(1,4),(2,5),(3,6)]
        #range(start,end,increment)
        training_batch = zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train)+1, batch_size))
        for start,end in training_batch:
            sess.run(optimizer, feed_dict={X:x_train[start:end], Y:y_train[start:end], p_keep_conv:0.8, p_keep_hidden:0.5})
        test_indicies = np.arange(len(x_test))
        np.random.shuffle(test_indicies)
        test_indicies = test_indicies[0:testing_size]
        print(i, np.mean(np.argmax(y_test[test_indicies], axis=1) ==
                         sess.run(predict_op, feed_dict={X: x_test[test_indicies],
                                                         Y: y_test[test_indicies],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
