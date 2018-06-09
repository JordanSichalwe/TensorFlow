import numpy as np
import tensorflow as tf
#import os, sys, inspect
from datetime import datetime
from matplotlib import pyplot as plt
import EmotionDetectorUtils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "EmotionDetector/", "path to data files")
tf.flags.DEFINE_string("logs_dir", "logs/EmotionDetector_logs/", "path to where logs are to be saved" )
tf.flags.DEFINE_string("mode", "train", "mode: train(Default)/test")

BATCH_SIZE=128
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 1001
#regularization is a process of introducing additional information
# in order to solve an ill-posed problem or to prevent overfitting.
REGULARIZATION = 1e-2
IMAGE_SIZE = 48
NUM_LABELS = 7
VALIDATION_PERCENT = 0.1

#WEIGTHTS
def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(intial)
    else:
        return tf.get_variable(name, initializer=initial)
weights = {
    'wc1':weight_variable([5,5,1,32], name="W_conv1"),
    'wc2':weight_variable([3,3,32,64], name="W_conv2"),
    'wf1':weight_variable([(IMAGE_SIZE//4)*(IMAGE_SIZE//4)*64,256], name="W_fc1"),
    'wf2':weight_variable([256,NUM_LABELS], name="W_fc2")
}
#BIAS
def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

biases = {
    'bc1':bias_variable([32],name="b_conv1"),
    'bc2':bias_variable([64],name="b_conv2"),
    'bf1':bias_variable([256],name="b_fc1"),
    'bf2':bias_variable([NUM_LABELS],name="b_fc2")

}
#LOSSES
def loss(pred, label):
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    reg_losses = tf.add_n(tf.get_collection("losses"))
    return cross_entropy_loss + REGULARIZATION * reg_losses

def add_to_regularization_loss(W, b):
    tf.add_to_collection("losses", tf.nn.l2_loss(W))
    tf.add_to_collection("losses", tf.nn.l2_loss(b))

def get_next_batch(images, labels, step):
    offset = (step * BATCH_SIZE) % (images.shape[0] - BATCH_SIZE)
    batch_images = images[offset: offset + BATCH_SIZE]
    batch_labels = labels[offset:offset + BATCH_SIZE]
    return batch_images, batch_labels

def emotion_cnn(dataset):
    with tf.name_scope("conv1") as scope:
        tf.summary.histogram("W_conv1",weights['wc1'])
        tf.summary.histogram("b_conv1",biases['bc1'])
        conv_1 = tf.nn.conv2d(dataset, weights['wc1'], strides=[1,1,1,1],padding="SAME")
        h_conv1 = tf.nn.bias_add(conv_1, biases['bc1'])
        h_1 = tf.nn.relu(h_conv1)
        h_pool1 =  tf.nn.max_pool(h_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        add_to_regularization_loss(weights['wc1'], biases['bc1'])
    with tf.name_scope('conv2') as scope:
        tf.summary.histogram("W_conv2", weights['wc2'])
        tf.summary.histogram("b_conv2", biases['bc2'])
        conv_2 = tf.nn.conv2d(h_pool1, weights['wc2'], strides=[1,1,1,1], padding="SAME")
        h_conv2 = tf.nn.bias_add(conv_2, biases['bc2'])
        h_2 = tf.nn.relu(h_conv2)
        h_pool2 = tf.nn.max_pool(h_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        add_to_regularization_loss(weights['wc2'], biases['bc2'])
    with tf.name_scope("fc_1") as scope:
        prob = 0.5
        image_size = IMAGE_SIZE//4
        h_flat = tf.reshape(h_pool2, [-1, image_size*image_size*64])
        tf.summary.histogram("W_fc1", weights['wf1'])
        tf.summary.histogram("b_fc1", biases['bf1'])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, weights['wf1'])+ biases['bf1'])
        h_fc1_dropout = tf.nn.dropout(h_fc1, prob)
    with tf.name_scope("fc_2") as scope:
        tf.summary.histogram("W_fc2", weights['wf2'])
        tf.summary.histogram("b_fc2", biases['bf2'])
        pred = tf.matmul(h_fc1_dropout, weights['wf2']) + biases['bf2']
    return pred

def main(argv=None):
    train_image, train_labels,valid_images, valid_labels, test_image = EmotionDetectorUtils.read_data(FLAGS.data_dir)
    print("train shape: %s "% train_image.shape[0])
    print("validation shape: %s"% valid_images.shape[0])
    print("Test size: %s" % test_image.shape[0])

    dropout_prob = tf.placeholder(tf.float32)
    input_dataset = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input")
    input_labels = tf.placeholder(tf.float32, [None, NUM_LABELS])
    global_step = tf.Variable(0,trainable=False)

    pred = emotion_cnn(input_dataset)
    output_pred = tf.nn.softmax(pred, name="output")
    loss_val = loss(pred, input_labels)

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_val)

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph_def)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model Restored")
        for step in range(MAX_ITERATIONS):
            batch_image, batch_label = get_next_batch(train_image, train_labels, step)
            feed_dict = {input_dataset:batch_image, input_labels:batch_label}
            sess.run(train_op, feed_dict=feed_dict)
            if step % 10 == 0:
                train_loss, summary_str = sess.run([loss_val, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
                print("Training Loss: %f" % train_loss)
            if step % 100 == 0:
                valid_loss = sess.run(loss_val, feed_dict={input_dataset: valid_images, input_labels: valid_labels})
                print("%s Validation Loss: %f" % (datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + 'model.ckpt', global_step=step)

if __name__ == "__main__":
    tf.app.run()
    """
    image_0 = train_image[0]
    label_0 = train_labels[0]

    image_0 = np.resize(image_0, (48, 48))
    plt.imshow(image_0, cmap='Greys_r')
    plt.show()"""

