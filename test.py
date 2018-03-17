import tensorflow as tf


# Hyperparameters
input_layer_size = 784
hidden_layer_size = 100
output_layer_size = 10

# Download MNIST dataset
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# Build the model
x = tf.placeholder(tf.float32, [None, input_layer_size], 'x')
y = tf.placeholder(tf.int32, [None], 'y')

hidden = tf.layers.dense(x, hidden_layer_size, activation=tf.nn.relu)
output = tf.layers.dense(hidden, output_layer_size)

pred = tf.argmax(output, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

# Checkpoint
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver.restore(sess, './logs/ckpt/model-5000.ckpt')

feed_dict = {x: mnist.test.images, y: mnist.test.labels}
test_accuracy = sess.run(acc, feed_dict)
print('\n\nTest accuracy:', test_accuracy)
