# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:57:58 2017

@author: Wayne
"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
learning_rate = 0.001
training_iters = 1000
dropout = 0.5
n_classes = 6
batch_size = 50
x = tf.placeholder(tf.float32,[None,8,8,4])
y = tf.placeholder(tf.float32,[None,6])
keep_prob = tf.placeholder(tf.float32)

def con2d(x,w,b):
     return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = 'SAME'),b))

def max_pool(EMG,k):
     return tf.nn.max_pool(EMG,ksize = [1,k,k,1],strides = [1,k,k,1],padding = 'SAME')

weights = {
          'wc1':tf.Variable(tf.random_normal([5,5,4,16])),
          'wc2':tf.Variable(tf.random_normal([5,5,16,64])),
          'wd1':tf.Variable(tf.random_normal([2*2*64,100])),
          'out':tf.Variable(tf.random_normal([100,n_classes]))
          }
bias = {
        'bc1':tf.Variable(tf.random_normal([16])),
        'bc2':tf.Variable(tf.random_normal([64])),
        'cd1':tf.Variable(tf.random_normal([100])),
        'out':tf.Variable(tf.random_normal([n_classes]))
        }
          

def conv_net(_x,_weights,_bias,_dropout):
     conv1 = con2d(_x,_weights['wc1'],_bias['bc1'])
     conv1 = max_pool(conv1,k=2)
     conv1 = tf.nn.dropout(conv1,_dropout)
     conv2 = con2d(conv1,_weights['wc2'],_bias['bc2'])
     conv2 = max_pool(conv2,k=2)
     conv2 = tf.nn.dropout(conv2,_dropout)
     
     full1 = tf.reshape(conv2,[-1,2*2*64])
     full2 = tf.nn.sigmoid(tf.add(tf.matmul(full1,_weights['wd1']),_bias['cd1']))
     full2 = tf.nn.dropout(full2,_dropout)
     out = tf.add(tf.matmul(full2,_weights['out']),_bias['out'])
     
     return out

prediction = conv_net(x,weights,bias,keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
train_accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

matDir = "G:\\电极位置\\吴乐\\mat\\"
trainFtr = sio.loadmat(matDir+'trainFtr')
train_d = trainFtr['trainFtr']
trainLabel = sio.loadmat(matDir+'trainLabel')
train_l = trainLabel['trainLabel']
train_l = train_l[0,:]-1
train_l = tf.one_hot(train_l,6,1,0)
train_l = sess.run(train_l)

testFtr = sio.loadmat(matDir+'testFtr')
test_d = testFtr['testFtr']
testLabel = sio.loadmat(matDir+'testlabel')
test_l = testLabel['testLabel']
test_l = test_l[0,:]-1
test_l = tf.one_hot(test_l,6,1,0)
test_l = sess.run(test_l)

m = int(train_d.shape[0]/batch_size)
for i in range(training_iters):
     for j in range(m):
          start = batch_size*j
          batch_d = train_d[start:start+batch_size,:,:,:]
          batch_l = train_l[start:start+batch_size,:]
#          batch_d = sess.run(batch_d)
#          batch_l = sess.run(batch_l)
          sess.run(optimizer,feed_dict={x:batch_d,y:batch_l,keep_prob:dropout})
     if i%50 == 0:
          correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
          accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
          print("step",i,"test accuracy:",sess.run(accuracy,feed_dict={x:test_d,y:test_l,keep_prob:dropout}))
          
     


