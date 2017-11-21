# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow.python.platform
# Import data
from tensorflow.examples.tutorials.mnist import input_data

# import input_data

import tensorflow as tf

import os

from mnist_demo import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data/', 'Directory for storing data')


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    sess = tf.InteractiveSession()

    # Create the model
    with tf.name_scope('input_data'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        W = tf.Variable(tf.zeros([784, 10]), name='weights')
        b = tf.Variable(tf.zeros([10]), name='bias')

    with tf.name_scope('Wx_b'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)

    # # Add summary ops to collect data
    _ = tf.summary.histogram('weights', W)
    _ = tf.summary.histogram('biases', b)
    _ = tf.summary.histogram('y', y)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('xent'):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        _ = tf.summary.scalar('cross entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Test trained model
    with tf.name_scope('test'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        _ = tf.summary.scalar('accuracy', accuracy)


        # Merge all the summaries and write them out to /tmp/mnist_logs

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/mnist-logs', sess.graph_def)
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()

    # Train
    # 判断模型保存路径是否存在，不存在就创建
    if not os.path.exists('softmax-model/'):
        os.mkdir('softmax-model/')

    # 初始化
    # sess = tf.Session()
    if os.path.exists('softmax-model/checkpoint'):  # 判断模型是否存在
        print("存在")
        saver.restore(sess, 'softmax-model/model.ckpt')  # 存在就从模型中恢复变量
        print("恢复")
        print(sess.run(W))
        print(sess.run(b))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("test accuracy %g" % accuracy.eval(
            feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        # sess.run([train_step], feed_dict={x: mnist.test.images[0:500], y_: mnist.test.labels[0:500]})
    else:
        init = tf.global_variables_initializer()  # 不存在就初始化变量
        print("初始化")
        sess.run(init)
    # print(tf.argmax(W,1).eval())
    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # print(tf.argmax(W,1).eval())
        train_step.run({x: batch_xs, y_: batch_ys})
        save_path = saver.save(sess, 'softmax-model/model.ckpt')
        rs = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys})
        writer.add_summary(rs, i)

        # break

    # accuracy=tf.cast(tf.argmax(y, 1), tf.float32)
    # accuracy=y
    # print((mnist.test.labels))
    print("test accuracy %g" % accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    sess.close()


if __name__ == '__main__':
    tf.app.run()
