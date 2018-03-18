import os
import shutil

import tensorflow as tf


# Hyperparameters
input_layer_size = 784
hidden_layer_size = 100
output_layer_size = 10
learning_rate = 0.1

num_iters = 5000
batch_size = 30

# Download MNIST dataset
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# Build the model
x = tf.placeholder(tf.float32, [None, input_layer_size], 'x')
y = tf.placeholder(tf.int32, [None], 'y')

hidden = tf.layers.dense(x, hidden_layer_size, activation=tf.nn.relu)
output = tf.layers.dense(hidden, output_layer_size)

pred = tf.argmax(output, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

loss = tf.losses.sparse_softmax_cross_entropy(y, output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

# Tensorboard
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', acc)
summ_op = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./logs/train')
valid_writer = tf.summary.FileWriter('./logs/valid')

# Checkpoint
saver = tf.train.Saver()

# Training iteration
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1, num_iters+1):
    batch_x, batch_y = mnist.train.next_batch(batch_size)

    fetch_dict = {'loss': loss, 'acc': acc, 'train': train_op,
                  'summary': summ_op}
    feed_dict = {x: batch_x, y: batch_y}

    result = sess.run(fetch_dict, feed_dict)

    train_writer.add_summary(result['summary'], i)

    # Validation
    if i % 50 == 0:
        valid_x = mnist.validation.images
        valid_y = mnist.validation.labels

        fetch_dict = {'loss': loss, 'acc': acc, 'summary': summ_op}
        feed_dict = {x: valid_x, y: valid_y}

        result = sess.run(fetch_dict, feed_dict)

        print('valid at step', i,
              'acc=%.3f, loss=%.3f' % (result['acc'], result['loss']))
        valid_writer.add_summary(result['summary'], i)
        saver.save(sess, './logs/ckpt/model-%d.ckpt' % (i))

print('Done!')
