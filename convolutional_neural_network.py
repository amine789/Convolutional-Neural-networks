# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:46:11 2019

@author: 1431145
"""

import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle

def y2indicator(y):
    N=len(y)
    ind = np.zeros((N,10))
    for i in range(N):
        ind[i,y[i]]=1
    return ind

def error_rate(p,t):
    return np.mean(p != t)

def conv2pool(x,w,b):
    convout = tf.nn.conv2d(x,w,strides=[1,1,1,1], padding="SAME")
    convout = tf.nn.bias_add(convout,b)
    conv2pool = tf.nn.max_pool(convout, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    return tf.nn.relu(conv2pool)
def init_filter(shape):
    w= np.random.randn(*shape)*np.sqrt(2/np.prod(shape[:-1]))
    return w.astype(np.float32)
def rearrange(x):
    
    N = x.shape[-1]
    out = np.zeros((N,32,32,3), dtype=np.float32)
    for i in range(N):
        for j in range(3):
            out[i,:,:,j] = x[:,:,j,i]
    return out/255

def main():
    train= loadmat("train_32x32.mat")
    test= loadmat("test_32x32.mat")
    xTrain = rearrange(train["X"])
    yTrain = train["y"].flatten() - 1
    print(len(yTrain))
    xTest = rearrange(test["X"])
    yTest = test["y"].flatten() - 1
    del train
    xTrain,yTrain =  shuffle(xTrain,yTrain)
    yTrain_ind = y2indicator(yTrain)
    del test
    yTest_ind = y2indicator(yTest)
    
    M=500
    batch_sz = 500
    N = len(xTrain)
    n_batches = N//500
    max_iter = 20
    
    K=10
    W1_shape = (5,5,3,20)
    W1_init = init_filter(W1_shape)
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32)
    W2_shape = (5,5,20,50)
    W2_init = init_filter(W2_shape)
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)
    W3_init = np.random.rand(W2_shape[-1]*8*8,M)/np.sqrt(W2_shape[-1]*8*8+M)
    b3_init = np.zeros(M,dtype=np.float32)
    W4_init = np.random.rand(M,K)/np.sqrt(M+K)
    b4_init = np.zeros(K, dtype=np.float32)
    X = tf.placeholder(tf.float32, shape=(batch_sz,32,32,3), name="X")
    Y = tf.placeholder(tf.float32, shape=(batch_sz,K), name="Y")
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))
    
    Z1 = conv2pool(X,W1,b1)
    Z2 = conv2pool(Z1,W2,b2)
    Z2_shape = Z2.get_shape().as_list()
    Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
    Z3 = tf.nn.relu(tf.matmul(Z2r,W3)+b3)
    Yish = tf.matmul(Z3,W4)+b4
    
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99,momentum=0.9).minimize(cost)
    predict_op = tf.argmax(Yish,1)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = xTrain[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = yTrain_ind[j*batch_sz:(j*batch_sz+batch_sz)]
                if len(Xbatch)==batch_sz:
                    sess.run(train_op, feed_dict={X:Xbatch,Y:Ybatch})
                    if j%50==0:
                        test_cost=0
                        prediction = np.zeros(len(yTest))
                        for k in range(len(xTest)//batch_sz):
                            Xbatch_test = xTest[k*batch_sz:(k*batch_sz+batch_sz)]
                            Ybatch_test = yTest_ind[k*batch_sz:(k*batch_sz+batch_sz)]
                            test_cost += sess.run(cost, feed_dict={X:Xbatch_test,Y:Ybatch_test})
                            prediction[k*batch_sz:(k*batch_sz+batch_sz)] = sess.run(predict_op, feed_dict={X:Xbatch_test})
                        err = error_rate(prediction,yTest)
                        print("cost/err at iteration i=%d, j=%d: %.3f /%.3f "%(i,j,test_cost,err))
                            
                        
                        
    
    