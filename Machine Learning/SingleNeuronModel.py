import tensorflow as tf
input = tf.constant(2.0,name="input")
weight = tf.Variable(1.0,name="weight")
expected_output = tf.constant(0.0,name="expected_output")
model = tf.multiply(input,weight,"model")
#losses
loss = tf.pow(model-expected_output,2,name="loss")
#optimizer
optim = tf.train.GradientDescentOptimizer(learning_rate=0.025).minimize(loss)
#summary for TensorBoard
for value in [input,weight,expected_output,loss]:
    summary = tf.summary.scalar(value.op.name,value)
Summaries = tf.summary.merge_all()
sess = tf.Session()
summary_writer = tf.summary.FileWriter('log_simple_stats',sess.graph)
sess.run(tf.global_variables_initializer())
for i in range(100):
    summary_writer.add_summary(sess.run(summary),i)
    sess.run(optim)
