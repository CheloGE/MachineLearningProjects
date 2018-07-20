import tensorflow as tf

# Create TensorFlow object called tensor
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

bias=tf.Variable(tf.zeros((n_labels,)))

tryChelo=tf.constant(2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output=sess.run([init,tryChelo])
    #output = sess.run(tryChelo)
    print(bias.eval())
    print(output)
    #print(weights.eval())
    