import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

logs_path = 'log_simple_stats_5_layers_sigmoid'
batch_size = 100
learning_rate = 0.5
training_epochs = 10
mnist = input_data.read_data_sets("data",one_hot=True)
#layer sizes
l1=1000
l2=500
l3=300
l4=150
l5=10
X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])
X_flat = tf.reshape(X, [-1, 784])

#layer1
W1 = tf.Variable(tf.truncated_normal([784,l1],stddev=0.1))
B1 = tf.Variable(tf.zeros([l1]))
Y1 = tf.nn.relu(tf.matmul(X_flat,W1)+B1)
Y1d = tf.nn.dropout(Y1,0.4)
#layer2
W2 = tf.Variable(tf.truncated_normal([l1,l2],stddev=0.1))
B2 = tf.Variable(tf.zeros([l2]))
Y2 = tf.nn.relu(tf.matmul(Y1d,W2)+B2)
Y2d = tf.nn.dropout(Y2,0.4)
#layer3
W3 = tf.Variable(tf.truncated_normal([l2,l3],stddev=0.1))
B3 = tf.Variable(tf.zeros([l3]))
Y3 = tf.nn.relu(tf.matmul(Y2d,W3)+B3)
Y3d = tf.nn.dropout(Y3,0.4)
#layer4
W4 = tf.Variable(tf.truncated_normal([l3,l4],stddev=0.1))
B4 = tf.Variable(tf.zeros([l4]))
Y4 = tf.nn.relu(tf.matmul(Y3d,W4)+B4)
Y4d = tf.nn.dropout(Y4,0.4)
#layer5
W5 = tf.Variable(tf.truncated_normal([l4,l5],stddev=0.1))
B5 = tf.Variable(tf.zeros([l5]))
Ylogits = tf.matmul(Y4d,W5)+B5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(learning_rate = 0.0005).minimize(cross_entropy)
#summaries
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples / batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, summary = sess.run([train_step, summary_op], feed_dict={X: batch_x, Y_: batch_y})
            writer.add_summary(summary, epoch * batch_count + i)
        print("Epoch: ", epoch)

    print("Accuracy:{}%".format(accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels})*100))
    print("done")

