# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:14:32 2020

@author: user
"""

from keras.models import Model
from keras.layers import Dense,Flatten,Input, Activation
from keras.applications import vgg16
from keras import backend as K
from keras import optimizers
from keras import metrics
from keras.utils.np_utils import to_categorical
from keras.optimizers import *
from keras import callbacks
from keras.callbacks import TensorBoard, CSVLogger
#--------------------------------------------------
# main net:
inputs = Input(shape=(512,))
x = Dense(300)(inputs)
x = Activation('relu')(x)
x = Dense(200)(x)
x = Activation('relu')(x)
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(512)(x)
out = Activation('linear')(x)
model = Model(inputs=inputs, outputs=out)
model.summary()

#train main net:
model.compile(loss='mean_squared_error',
                  optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06),
                  metrics=[metrics.mean_squared_error])
    
num_examples = X_train.shape[0]
num_epochs = 20
batch_size = 512
num_batch_per_epoch = num_examples / batch_size
model.fit(x=X_train,y=Y_train,batch_size=batch_size,epochs=num_epochs)
#--------------------------------------------------



#--------------------------------------------------
#deleting last layer
model.layers.pop()
model.layers.pop()
model.summary()

#adding new layers
x = Dense(600)(model.layers[-1].output)
x = Activation('relu')(x)
x = Dense(300)(x)
out = Activation('softmax')(x)

model2 = Model(inputs=inputs, outputs=out)
model2.summary()

#train second net:
# how not to strart from random?:
# answer:  we shouldn't compile the model again as compiling will initialize model from random 
#model2.compile !!!!!!!!!!don't use it
model2.fit
#--------------------------------------------------


























