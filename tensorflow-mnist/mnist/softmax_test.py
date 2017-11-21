#!/usr/bin/env python
# encoding: utf-8

"""
@author: qli
@file: softmax_test.py
@time: 17-4-27 下午3:17
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# Import data
# from tensorflow.examples.tutorials.mnist import input_data

import input_data

import tensorflow as tf

import os

from mnist_demo import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
sess = tf.InteractiveSession()
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
    #sess.run([train_step], feed_dict={x: mnist.test.images[0:500], y_: mnist.test.labels[0:500]})
else:
    init = tf.global_variables_initializer()  # 不存在就初始化变量
    sess.run(init)
# print(tf.argmax(W,1).eval())

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy=tf.cast(tf.argmax(y, 1), tf.float32)
# accuracy=y
# print((mnist.test.labels))

dir_name = "test_num"
files = os.listdir(dir_name)
cnt=len(files)
for i in range(cnt):
  files[i]=dir_name+"/"+files[i]
  print(files[i])
  test_images1,test_labels1=GetImage([files[i]])
  print (tf.cast(correct_prediction, tf.float32).eval)
  print(shape(test_images1))
  mnist.test = input_data.DataSet(test_images1, test_labels1, dtype=tf.float32)
  res=accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
  print("hello")
  print(res)
  # print(shape(mnist.test.images))
  # print (tf.argmax(y, 1))
  # print(y.eval())
  print("output:",int(res))
  print("\n")
  # if(res==1):
  #   print("correct!\n")
  # else:
  #   print("wrong!\n")

  # print("input:",files[i].strip().split('/')[1][0])

# print(print_tensors_in_checkpoint_file('tmp/model.ckpt',None,True))

