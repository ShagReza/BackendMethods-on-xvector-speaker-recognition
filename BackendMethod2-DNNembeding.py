# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:50:19 2020

@author: user
"""

#------------------------------------------------------------------------------
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras import optimizers
from keras import metrics
from utils import load_data
import scipy.io
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.optimizers import *
from keras import callbacks
from keras.callbacks import TensorBoard, CSVLogger
import os
from sklearn.model_selection import train_test_split
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#load data
import numpy as np
filename='test_kermanshah_xvec.npz'
with np.load(filename , allow_pickle=True) as da:
    test_name = da['data_name']
    x_test = da['features']
    
    
filename='xvectors.npz'
    
with np.load(filename , allow_pickle=True) as da:
    train_name = da['spk_name']
    x_train= da['features']
    
label_train = pd.factorize(train_name)[0]
Num_spks_train=np.max(label_train)+1
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#norm
dim_lda=512
norm_X_test = np.linalg.norm(x_test,axis=1)
B=np.transpose(np.tile(norm_X_test, (dim_lda,1)))
x_test=np.divide(x_test,B)
norm_X_train = np.linalg.norm(x_train,axis=1)
B=np.transpose(np.tile(norm_X_train, (dim_lda,1)))
x_train=np.divide(x_train,B)
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#Finding Spks with just one file
IndexOne=[0]
for i in range(len(x_train)):
        index=np.where(label_train==i)[0]
        if len(index)==1:
            IndexOne=np.concatenate((IndexOne,index),axis=0)
IndexOne=list(IndexOne)
del (IndexOne[0])

#removing IndexOne
label_train2=list(label_train)
#label_train3= [x for x in label_train2 if x not in IndexOne]  #removing a list from another one by value
label_train2 = [i for j, i in enumerate(label_train2) if j not in IndexOne] #removing a list from another one by index
x_train2=x_train[label_train2]

#note by doing so, around 512 data having just one sample will be removed from backend methods
X_train3,X_val,Y_train3,Y_val=train_test_split(x_train2,label_train2, test_size=0.3, random_state=0,stratify=label_train2)
Y_train4 = to_categorical(Y_train3)


#A=np.asarray(label_train2) #list to array 

#deleting zero columns for train part
A=np.sum(Y_train4,axis=0)   
IndexOne=np.where(A==0)[0]     
IndexAll=np.asarray(list(range(3891)))
IndexNotOne=np.asarray( [i for j, i in enumerate(IndexAll) if j not in IndexOne])
Y_train4=np.transpose(Y_train4)
Y_train=np.transpose(Y_train4[IndexNotOne])
X_train=X_train3

#deleting zero columns for val part
Y_val = to_categorical(Y_val)
Y_val=np.transpose(Y_val)
Y_val=np.transpose(Y_val[IndexNotOne])
#------------------------------------------------------------------------------


#----------------------------------------------------------------------------
                            # without pretraining:      
OutputLen=3359
InputLen=512
#--------------
inputs = Input(shape=(InputLen,))
x = Dense(300)(inputs)
x = Activation('relu')(x)
x = Dense(200)(x)
x = Activation('relu')(x)
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dense(600)(x)
x = Activation('relu')(x)
x = Dense(OutputLen)(x)
out = Activation('softmax')(x)
model = Model(inputs=inputs, outputs=out)
model.summary()
#------------------------
opt1 = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt1, loss='categorical_crossentropy', metrics= ['accuracy'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=7, verbose=1, mode='max', epsilon=0.0001, cooldown=0, min_lr=0)
folder_log='.\log'
folder_model='.\model'
tensorboard1 = TensorBoard(log_dir = folder_log)
logger = CSVLogger(folder_log + '/training.log')
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}_trn_{acc:.4f}_val_{val_acc:.4f}.hdf5')
model_checkpoint = callbacks.ModelCheckpoint(
    						filepath= path_model,
    						monitor="val_acc",
    						mode="max",
    						verbose=0,
    						save_best_only=False,
                            save_weights_only=False        
                            )    
#------------------------
history = model.fit(X_train, Y_train, 
                    batch_size=128, 
                    epochs=50,
                    validation_data=(X_val, Y_val),
                    shuffle=True,
                    verbose=1,
                    callbacks= [
                        model_checkpoint
                        , tensorboard1
                        , logger
                        , reduce_LR
                        ]
                    )
#------------------------------------------------------------------------------
from keras.models import load_model
model = load_model('.\model\model_epoch_14_trn_0.8935_val_0.9216.hdf5')
model.summary()
# to get embeding layer's output
new_model = Model(inputs=model.input,outputs=model.get_layer("activation_17").output)

#DAE predict
x_test =new_model.predict(x_test,batch_size=512) 
x_train =new_model.predict(x_train,batch_size=512) 
#norm
dim_lda=600
norm_X_test = np.linalg.norm(x_test,axis=1)
B=np.transpose(np.tile(norm_X_test, (dim_lda,1)))
x_test=np.divide(x_test,B)
norm_X_train = np.linalg.norm(x_train,axis=1)
B=np.transpose(np.tile(norm_X_train, (dim_lda,1)))
x_train=np.divide(x_train,B)

#lda
dim_lda=150;
lda = LinearDiscriminantAnalysis(n_components=dim_lda)
lda.fit(x_train, label_train)
x_test=lda.transform(x_test)
#norm
norm_X_test = np.linalg.norm(x_test,axis=1)
B=np.transpose(np.tile(norm_X_test, (dim_lda,1)))
x_test=np.divide(x_test,B)
#
x_test_filtered=x_test
D2 = scipy.io.loadmat('test_ivecs.mat') 
D2['Results']['ivectors'][0,0]=np.transpose(x_test_filtered)
scipy.io.savemat('test_ivecs_filtered.mat', D2)















##------------------------------------------------------------------------------
#                     #with preprocessing
## main net:
#inputs = Input(shape=(512,))
#x = Dense(300)(inputs)
#x = Activation('relu')(x)
#x = Dense(200)(x)
#x = Activation('relu')(x)
#x = Dense(300)(x)
#x = Activation('relu')(x)
#x = Dense(512)(x)
#out = Activation('linear')(x)
#model = Model(inputs=inputs, outputs=out)
#model.summary()
##--------------
##train main net:
#model.compile(loss='mean_squared_error',
#                  optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06),
#                  metrics=[metrics.mean_squared_error])
##add callbacks!!!!!!!!!!!!!!!!    
#num_examples = x_train.shape[0]
#num_epochs = 20
#batch_size = 512
#num_batch_per_epoch = num_examples / batch_size
#model.fit(x=x_train,y=x_train,batch_size=batch_size,epochs=num_epochs)
##--------------
##deleting last layer
#model.layers.pop()
#model.layers.pop()
#model.summary()
##adding new layers
#x = Dense(600)(model.layers[-1].output)
#x = Activation('relu')(x)
#x = Dense(300)(x)
#out = Activation('softmax')(x)
#
#model2 = Model(inputs=inputs, outputs=out)
#model2.summary()
##train second net:
## how not to strart from random?:
## answer:  we shouldn't compile the model again as compiling will initialize model from random 
##model2.compile !!!!!!!!!!don't use it
#model2.fit
##------------------------------------------------------------------------------
#
#




#
##--------------
#model = load_model('.\model\model_epoch_14_trn_0.8935_val_0.9216.hdf5')
##--------------
#inputs = Input(shape=(InputLen,))
#x = Dense(300)(inputs)
#x = Activation('relu')(x)
#x = Dense(200)(x)
#x = Activation('relu')(x)
#x = Dense(300)(x)
#x = Activation('relu')(x)
#x = Dense(512)(x)
#x = Activation('relu')(x)
#x = Dense(600)(x)
#x = Activation('relu')(x)
#x = Dense(OutputLen)(x)
#out = Activation('softmax')(x)
#model2 = Model(inputs=inputs, outputs=out)
##--------------
#model2.layers[0].set_weights(model.layers[0].get_weights())
#model2.layers[1].set_weights(model.layers[1].get_weights())
#model2.layers[2].set_weights(model.layers[2].get_weights())
#model2.layers[2].set_weights(model.layers[2].get_weights())
#model2.layers[2].set_weights(model.layers[2].get_weights())
#model2.layers[12].set_weights(model.layers[12].get_weights())

