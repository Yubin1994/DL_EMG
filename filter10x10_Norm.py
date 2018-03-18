#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:40:54 2018

@author: wule
"""

import scipy.io as sio  
from scipy import stats
from keras.models import Model,load_model
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import regularizers

import keras.backend as K
K.set_image_data_format('channels_last')

path='G:\\吴乐\\包络18_03_07\\滤波\\leftTop_50ms_Norm\\7cls\\data_test'
data=sio.loadmat(path)
X_test = data['data_test']
X_test = X_test[...,np.newaxis] #增加维度

path='G:\\吴乐\\包络18_03_07\\滤波\\leftTop_50ms_Norm\\7cls\\label_test'
data=sio.loadmat(path)
label_test = data['label_test']

def SEMGModel(input_shape):
    """
    Implementation of the Model.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(input_shape)
    
    print(X_input.shape) #8 8 
    
    
    X = Conv2D(32, (3, 3), strides = (1, 1), padding = 'valid',name = 'conv1',kernel_regularizer=regularizers.l2(0.05))(X_input) # 6 
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    print(X.shape)
#    X = Dropout(0.3)(X)
    
#    X = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same',name = 'conv2',kernel_regularizer=regularizers.l2(0.01))(X) #4 4
#    X = BatchNormalization(axis = 3, name = 'bn2')(X)
#    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2),strides = (1, 1), name='max_pool1')(X) # 5 5
    print(X.shape)
    
    
    X = Conv2D(64, (3, 3), strides = (1, 1), padding = 'valid',name = 'conv2',kernel_regularizer=regularizers.l2(0.05))(X) #3 3
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    print(X.shape)
    
    X = Conv2D(128, (3, 3), strides = (1, 1), padding = 'valid',name = 'conv3',kernel_regularizer=regularizers.l2(0.05))(X) #1 1
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    print(X.shape)
    
#    X = Conv2D(128, (3, 3), strides = (1, 1), padding = 'same',name = 'conv4',kernel_regularizer=regularizers.l2(0.01))(X) #4 4
#    X = BatchNormalization(axis = 3, name = 'bn4')(X)
#    X = Activation('relu')(X)

    # convert fc connection to the convnet

    X = Conv2D(256,(1,1),strides = (1, 1),name='fc1',kernel_regularizer=regularizers.l2(0.01))(X)
    X = Conv2D(64,(1,1),strides = (1, 1),name='fc2',kernel_regularizer=regularizers.l2(0.01))(X)
    X = Conv2D(7,(1,1),strides = (1, 1),name='fc3',kernel_regularizer=regularizers.l2(0.01))(X) # 1*1*11
    X = Activation('softmax')(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='SEMGModel')
    
    ### END CODE HERE ###
    
    return model

SemgModel = SEMGModel((10,10,1))
SemgModel.load_weights('SemgModel_7cls_8x8_50ms.h5',by_name = True)

result = SemgModel.predict(X_test) #得到结果


#cls = 7;
#predsum = np.sum(result,axis=1)
#predsum = np.sum(predsum,axis=1)
#lengthY,lengthRow = predsum.shape
#pred = np.argmax(predsum , axis=1)
#ans=0
#for li in range(lengthY):
#     ans+=(pred[li]==label_test[li])*1
#
#accplot = ans/lengthY


cls = 7;
pred = np.argmax(result , axis=3) #取概率最大值作为预测结果
lengthY, lengthRow, lengthColumn = pred.shape
predbool = np.zeros(pred.shape) #是否预测正确正确

for li in range (lengthY):
    for lr in range (lengthRow):
        for lc in range (lengthColumn):
            predbool[li,lr,lc] = (pred[li,lr,lc]==label_test[li])*1
            

#predbool2 = predbool.reshape(lengthRow, lengthColumn,lengthY)
predsum = np.sum(predbool,axis=0)
predplot = predsum/lengthY*100

maxRow = 1;maxColumn=2;
accpred = pred[:,maxRow,maxColumn]
accMatrix = np.zeros([cls,cls])
for i in range(lengthY):
    accMatrix[label_test[i], accpred[i]]+= 1
sa = np.sum(accMatrix,axis = 1,keepdims=True)
sar = np.repeat(sa,cls,axis=1)
accPlot =  accMatrix/sar




##按照概率最大来看
#cls = 7
#pred = np.zeros(label_test.shape, dtype=np.int8)
#lengthY = pred.shape[0]
#for i in range(lengthY):
#    pred[i] = np.argmax(result[i]) % cls
#accMatrix = np.zeros([cls,cls])
#for i in range(lengthY):
#    accMatrix[label_test[i], pred[i]]+= 1
#sa = np.sum(accMatrix,axis = 1,keepdims=True)
#sar = np.repeat(sa,cls,axis=1)
#accPlot =  accMatrix/sar
#
#correctRes = 0;
#for i in range(cls):
#    correctRes += accMatrix[i,i]
#overallAcc = correctRes/lengthY