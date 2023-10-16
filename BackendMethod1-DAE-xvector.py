# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:39:21 2020

@author: user
"""

#---------------------------------------------------
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
#---------------------------------------------------


#load data-----------------------------------------
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
#-------------------------------------------


method=4

if method==1:
    #-------------------------------------------
    #Centralization
    Num_train=len(x_train)
    Num_test=len(x_test)
    mean_train=np.mean(x_train,axis=0)
    mean_matrix_train=np.tile(mean_train, (Num_train,1))
    x_train=x_train-mean_matrix_train
    mean_matrix_test=np.tile(mean_train, (Num_test,1))
    x_test=x_test-mean_matrix_test
    #-------------------------------------------
    #LDA
    dim_lda=150;
    lda = LinearDiscriminantAnalysis(n_components=dim_lda)
    lda.fit(x_train, label_train)
    x_test_filtered=lda.transform(x_test)
    #-------------------------------------------
    D2 = scipy.io.loadmat('test_ivecs.mat') 
    D2['Results']['ivectors'][0,0]=np.transpose(x_test_filtered)
    scipy.io.savemat('test_ivecs_filtered.mat', D2)
    #-------------------------------------------


if method==2:
        dim=512
    #-------------------------------------------
    #norm
    norm_x_test = np.linalg.norm(x_test,axis=1)
    B=np.transpose(np.tile(norm_x_test, (dim,1)))
    x_test=np.divide(x_test,B)
    norm_x_train= np.linalg.norm(x_train,axis=1)
    B=np.transpose(np.tile(norm_x_train, (dim,1)))
    x_train=np.divide(x_train,B)
    #-------------------------------------------
    #Centralization

    Num_train=len(x_train)
    Num_test=len(x_test)
    mean_train=np.mean(x_train,axis=0)
    mean_matrix_train=np.tile(mean_train, (Num_train,1))
    x_train=x_train-mean_matrix_train
    mean_matrix_test=np.tile(mean_train, (Num_test,1))
    x_test=x_test-mean_matrix_test
    
    #-------------------------------------------
    #LDA
    dim_lda=150;
    lda = LinearDiscriminantAnalysis(n_components=dim_lda)
    lda.fit(x_train, label_train)
    x_test=lda.transform(x_test)
    x_train=lda.transform(x_train)
    #------------------------------------------
    #norm
    norm_x_test = np.linalg.norm(x_test,axis=1)
    B=np.transpose(np.tile(norm_x_test, (dim_lda,1)))
    x_test=np.divide(x_test,B)
    norm_x_train= np.linalg.norm(x_train,axis=1)
    B=np.transpose(np.tile(norm_x_train, (dim_lda,1)))
    x_train=np.divide(x_train,B)
    #-------------------------------------------
    x_test_filtered=x_test
    D2 = scipy.io.loadmat('test_ivecs.mat') 
    D2['Results']['ivectors'][0,0]=np.transpose(x_test_filtered)
    scipy.io.savemat('test_ivecs_filtered.mat', D2)
    #-------------------------------------------


#------------------------------------------------------------
# Denoising autoencoder
def DAE(iv_dim,X_train,Y_train):    
    nu=2000
    #iv_dim = 150
    inputs = Input(shape=(iv_dim,))
    x = Dense(nu)(inputs)
    x = Activation('tanh')(x)
    x = Dense(iv_dim)(x)
    out = Activation('linear')(x)
    model = Model(inputs=inputs, outputs=out)
    # Set random seed to make results reproducible
    seed = 134
    np.random.seed(seed)
    # DAE training
    model.compile(loss='mean_squared_error',
                  optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06),
                  metrics=[metrics.mean_squared_error])
    
    num_examples = X_train.shape[0]
    num_epochs = 20
    batch_size = 512
    num_batch_per_epoch = num_examples / batch_size
    model.fit(x=X_train,y=Y_train,batch_size=batch_size,epochs=num_epochs)
    return(model)
#------------------------------------------------------------







#---------------------------------------------------
    # norm+lda+norm+(mean+norm)+DAE+norm+LDA+norm
if method==4:
    #lda
    dim_lda=150;
    lda = LinearDiscriminantAnalysis(n_components=dim_lda)
    lda.fit(x_train, label_train)
    x_test=lda.transform(x_test)
    x_train=lda.transform(x_train)
    #norm
    norm_x_test = np.linalg.norm(x_test,axis=1)
    B=np.transpose(np.tile(norm_x_test, (dim_lda,1)))
    x_test=np.divide(x_test,B)
    norm_x_train= np.linalg.norm(x_train,axis=1)
    B=np.transpose(np.tile(norm_x_train, (dim_lda,1)))
    x_train=np.divide(x_train,B)
    #compute mean vector
    s = (len(x_train),len(x_train[0]))
    meanTrain=np.zeros(s)
    for i in range(len(x_train)):
        index=np.where(label_train==i+1)[0]
        mean=np.mean(x_train[index,:],axis=0)
        #norm
        norm_mean = np.linalg.norm(mean)
        mean=mean/norm_mean
        #
        meanTrain[index,:]=np.tile(mean, (len(index),1))
    #DAE+norm
    Y_train=meanTrain;
    model=DAE(dim_lda, x_train, Y_train)
    #DAE predict
    x_test = model.predict(x_test,batch_size=512) 
    x_train = model.predict(x_train,batch_size=512) 
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