#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:09:13 2018

@author: wule
"""

import scipy.io as sio  
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model,load_model
from keras import regularizers
from keras import optimizers

import keras.backend as K
K.set_image_data_format('channels_last')

path='G:\\吴乐\\包络18_03_07\\滤波\\baseline_50ms_8x8_Norm\\7cls\\data_train'
data=sio.loadmat(path)
X_train = data['data_train']
X_train = X_train[...,np.newaxis]

path='G:\\吴乐\\包络18_03_07\\滤波\\baseline_50ms_8x8_Norm\\7cls\\label_train'
data=sio.loadmat(path)
label_train = ((data['label_train']).T )

path='G:\\吴乐\\包络18_03_07\\滤波\\baseline_50ms_8x8_Norm\\7cls\\data_test'
data=sio.loadmat(path)
X_test = data['data_test']
X_test = X_test[...,np.newaxis]

path='G:\\吴乐\\包络18_03_07\\滤波\\baseline_50ms_8x8_Norm\\7cls\\label_test'
data=sio.loadmat(path)
label_test = ((data['label_test']).T )


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

clsnum = 7
Y_train_temp = (convert_to_one_hot(label_train, clsnum)).T
Y_test_temp = (convert_to_one_hot(label_test, clsnum)).T
Y_train = Y_train_temp.reshape(Y_train_temp.shape[0],1,1,Y_train_temp.shape[1])
Y_test = Y_test_temp.reshape(Y_test_temp.shape[0],1,1,Y_test_temp.shape[1])

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

SemgModel = SEMGModel((8,8,1))
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
SemgModel.compile(optimizer =adam, loss ="categorical_crossentropy",metrics =["accuracy"])
SemgModel.fit(x=X_train,y=Y_train,epochs=10,batch_size=64,shuffle=True,validation_split=0.1)

## 用于保存参数
SemgModel.save_weights('SemgModel_7cls_8x8_50ms.h5')
#SemgModel.load_weights('SemgModel_6cls_8x8_50ms.h5',by_name = True)
preds = SemgModel.evaluate(x =X_test,y =Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
    
#t = X_test[0:1]
#
#result = SemgModel.predict(t)
#r2 = result.reshape(1,8)
