# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 00:14:08 2019
@author: amine bahlouli
"""




import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np

def init_weight_and_bias(M1,M2):
    W = np.random.rand(M1,M2)/np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def init_filter(shape):
    w = np.random.rand(*shape)*np.sqrt(2/np.prod(shape[:-1]))
    return w.astype(np.float32)
def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind
def getData(balance_ones=True):
    Y=[]
    X=[]
    first=True
    for line in open("fer2013.csv"):
        if first:
            first=False
        else:
            row = line.split(",")
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
    X,Y = np.array(X)/255.0, np.array(Y)
    if balance_ones:
        X0, Y0 = X[Y!=1], Y[Y!=1]
        X1 = X[Y==1]
        X1 = np.repeat(X1,9,axis=0)
        X = np.vstack([X0,X1])
        Y = np.concatenate((Y0, [1]*len(X1)))
    return X,Y
def getImageData():
    X,Y = getData()
    N,D = X.shape
    d = int(np.sqrt(D))
    X= X.reshape(N,1,d,d)
    return X,Y
class HiddenLayer:
    def __init__(self,M1,M2,id):
        self.M1=M1
        self.M2=M2
        self.id=id
        W,b = init_weight_and_bias(M1,M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W,self.b]
    def forward(self,x):
        return tf.nn.relu(tf.matmul(x,self.W)+self.b)
class ConvPoolLayer:
    def __init__(self,mi,mo,fw=5,fh=5,poolsz=(2,2)):
        sz= (fw,fh,mi,mo)
        W0 = init_filter(sz)
        self.W = tf.Variable(W0)
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = tf.Variable(b0)
        self.poolsz=poolsz
        self.params = [self.W,self.b]
    def forward(self,x):
        convout = tf.nn.conv2d(x,self.W,srides=[1,1,1,1], padding="SAME")
        convout = tf.nn.bias_add(convout,self.b)
        pool_out = tf.nn.max_pool(convout, ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")
        return tf.tanh(pool_out)
class CNN:
    def __init__(convpool_layer_size,hiden_layer_size):
        self.convpool_layer_size=convpool_layer_size
        self.hidden_layer_size=hidden_layer_size
    def fit(self, X,Y,lr=10e-4,mu=0.99,reg=10e-4,decay=0.9999,eps=10e-3,batch_sz=30,epochs=3,show_fig=True):
        lr = np.float32(lr)
        mu = np.np.float32(mu)
        reg = np.float32(reg)
        decay = np.float32(decay)
        eps = np.float32(eps)
        K = len(set(Y))
        X,Y = shuffle(X,Y)
        X = np.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)
        xValid, yValid = X[-1000:],Y[-1000:]
        X,Y = X[:-1000,],Y[:-1000,]
        yValid_flat= np.argmax(yValid,axis=1)
        N,d,d,c = X.shape
        mi=c
        outW=d
        outH=d
        self.convpool_layer=[]
        for mo,fw,fh in self.convpool_layer_size:
            layer = convPoolLayer(mi,mo,fw,fh)
            self.convpool_layer.append(layer)
            outW=outW/2
            outH = outH/2
            mi=mo
        self.hidden_layer=[]
        M1 = self.convpool_layer_size[-1][0]*outW*outH
        for M2 in self.hidden_layer_size:
            h = HiddenLayer(M1,M2,count)
            self.hidden_layer.append(h)
            M1=M2
            count+=1
