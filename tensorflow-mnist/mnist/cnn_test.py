#!/usr/bin/env python
# encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
#from tensorflow.examples.tutorials.mnist import input_data

import input_data
import numpy as np

import tensorflow as tf

import os

from mnist_demo import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)  # 截断正态分布，此函数原型为尺寸、均值、标准差
    return tf.Variable(initial)


def bias_variable(shape, name):  #产生随机变量
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def conv2d(x, W):
    # 步长=[1,x_movement,y_movement,1] padding=same  抽取有一部分在图片外面,以0填充;抽取出来的长和宽与图片一致     padding=valid抽取的比图片小；抽取出来比图片小
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长   二维神经网络


def max_pool_2x2(x):
    # 为防止跨度太大，丢失东西太多，添加pooling-》跨度减小，用pooling增大跨度    [batch, height, width, channels]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 参数同上，ksize是池化块的大小


with tf.name_scope('input'):
    x = tf.placeholder("float", shape=[None, 784], name='x-input')   # 28*28
    y_ = tf.placeholder("float", shape=[None, 10], name='y-input')
    # 我们还定义了dropout的placeholder，它是解决过拟合的有效手段
    keep_prob = tf.placeholder(tf.float32, name='keep-pro')
    tf.summary.scalar('dropout_keep_pro', keep_prob)

with tf.name_scope('model'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])   # -1代表先不考虑输入的图片例子多少这个维度，   28,28像素点   channel=1，黑白
    tf.summary.image('input1', x_image)

    with tf.name_scope('layer1'):
        w_conv1 = weight_variable([5, 5, 1, 32],  # patch 5*5，被收集原图的像素 ，in size 1，out size 32  收集一个单位，高32
                                  'weight1')  # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征
        b_conv1 = bias_variable([32], 'bias1')
        tf.summary.histogram('weight1', w_conv1)
        tf.summary.histogram('bias1', b_conv1)
        with tf.name_scope('conv1'):
            h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)   # 非线性处理  原图像以5×5的像素抽出，抽出的高度变为32，所以output 28*28*32
        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)   # output 14*14*32

    with tf.name_scope('layer2'):
        w_conv2 = weight_variable([5, 5, 32, 64], 'weight2')  # 多通道卷积，卷积出64个特征 patch 5*5，in size 32，out size 64
        b_conv2 = bias_variable([64], 'bias2')
        tf.summary.histogram('weight2', w_conv2)
        tf.summary.histogram('bias2', b_conv2)
        with tf.name_scope('conv2'):
            h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)  # output 14*14*64
        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)  # output 7×7*64

    # Define loss and optimizer
    with tf.name_scope('full_connect1'):
        # 我们通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出结果展平.
        w_fc1 = weight_variable([7 * 7 * 64, 1024], 'weight3')
        b_fc1 = bias_variable([1024], 'bias3')
        tf.summary.histogram('weight3', w_fc1)
        tf.summary.histogram('bias3', b_fc1)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 展开，第一个参数为样本数量，-1未知    将上一个输出结果展平  三维变一维
        f_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    with tf.name_scope('full_connect2'):
        # 为防止过拟合
        h_fc1_drop = tf.nn.dropout(f_fc1, keep_prob)
        # output
        w_fc2 = weight_variable([1024, 10], 'weight4')  # 输入1024，输出10
        b_fc2 = bias_variable([10], 'bias4')
        tf.summary.histogram('weight4', w_fc2)
        tf.summary.histogram('bias4', b_fc2)
    with tf.name_scope('soft-max'):
        # 用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

with tf.name_scope('result'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 定义交叉熵为loss函数

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 调用优化器优化
    # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accurancy')
    # accuracy = tf.cast(tf.argmax(y_conv, 1), tf.float32, name='accurancy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

sess = tf.InteractiveSession()
summary_op = tf.summary.merge_all()

sess.run(tf.initialize_all_variables())
accuracy1 = tf.cast(tf.argmax(y_conv, 1), tf.float32)
# 模型保存加载工具
saver = tf.train.Saver()

# 判断模型保存路径是否存在，不存在就创建
if not os.path.exists('cnn-model/'):
    os.mkdir('cnn-model/')

# 初始化
# sess = tf.Session()
if os.path.exists('cnn-model/checkpoint'):  # 判断模型是否存在
    print("存在")
    saver.restore(sess, 'cnn-model/model.ckpt')  # 存在就从模型中恢复变量
    print("恢复")
    # sess.run([train_step], feed_dict={x: mnist.test.images[0:500], y_: mnist.test.labels[0:500], keep_prob: 1.0})
else:
    init = tf.global_variables_initializer()  # 不存在就初始化变量
    sess.run(init)


# Train
# tf.initialize_all_variables().run()

# for i in range(2000):
#     batch = mnist.train.next_batch(50)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#     if i % 100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
#         print("step %d, training accuracy %g" % (i, train_accuracy))
#     #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 30.5})
#     save_path = saver.save(sess, 'cnn-model/model.ckpt')
# print(tf.argmax(W,1).eval())
#LALALA
#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images[0:500], y_: mnist.test.labels[0:500], keep_prob: 1.0}))

    # break

# Test trained model
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#accuracy = tf.cast(tf.argmax(y_conv, 1), tf.float32)
# accuracy=y
# print((mnist.test.labels))

dir_name = "test_num"
files = os.listdir(dir_name)
cnt = len(files)
for i in range(cnt):
    files[i] = dir_name + "/" + files[i]
    # print(files[i])
    test_images1, test_labels1 = GetImage([files[i]])
    # print (tf.cast(correct_prediction, tf.float32).eval)
    # print(shape(test_images1))
    mnist.test = input_data.DataSet(test_images1, test_labels1, dtype=tf.float32)
    res = accuracy1.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

    # print(shape(mnist.test.images))
    # print (tf.argmax(y, 1))
    # print(y.eval())
    print("output:", int(res))
    print("\n")
    # if(res==1):
    #   print("correct!\n")
    # else:
    #   print("wrong!\n")

    # print("input:",files[i].strip().split('/')[1][0])
