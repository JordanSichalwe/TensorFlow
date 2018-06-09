from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#imports
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


#function, which conforms to the interface expected by TensorFlow's Estimator API
def cnn_model_fn(features, labels, mode):
    #model function fo CNN|returns prdictions,loss and training operation
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    #convolutional layer #1
    conv_layer1 = tf.layers.conv2d(
        #input
        inputs = input_layer,
        #number of filters
        filters = 32,
        #filter_kernel_size
        kernel_size=[5, 5],
        padding="same",
        #activation function = ReLU(rectified Linear Unit):ensures nonlinearities
        activation=tf.nn.relu
    )
    #pooling layer#1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv_layer1,
        pool_size=[2, 2],
        #subregions extracted by the filter should be separated by 2 pixels
        # in both the width and height dimensions
        # (for a 2x2 filter, this means that none of the regions extracted will overlap).
        strides=2
    )
    #convolutiona layer #2 and pooling layer#2
    conv_layer2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv_layer2, pool_size=[2, 2], strides=2)
    #Dense layer#1 with a droupout
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense1 = tf.layers.dense(
        inputs=pool2_flat,
        #number of nodes
        units=1024,
        #activation function
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense1,
        #probability of drop(regularitization rate)
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    #dense#2(logits layer)
    dense_logits = tf.layers.dense(
        inputs=dropout,
        units=10
    )
    #create a python dictonary for predictions
    predictions = {
        #Generate predictions (for PREDICT and EVAL modes)
        "classes":tf.argmax(input=dense_logits, axis=1),
        #Add 'softmax_tensor' to the graph.It is used for PREDICT and by th 'logging_hook
        "probabilites":tf.nn.softmax(dense_logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    #calculation of loss(For both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=dense_logits)

    #configure the TRAINING operation(for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
def main(unused_argv):
    #Loading training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images #Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images #Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="tmp\mnist_convnet_model")
    # Set up logging for predictions
    tensors_to_log = {"probabilities":"softmax_tensor"}
    loggiing_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[loggiing_hook]
    )
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


#our application logic will be added here
#direct running as a main method(check)
if __name__ == '__main__':
    tf.app.run()