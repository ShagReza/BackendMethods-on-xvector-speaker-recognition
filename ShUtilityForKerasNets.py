
##---------------------------------------------------------------------
## To check wheter GPU is available:
## 1:
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
## 2:
#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
##---------------------------------------------------------------------





##---------------------------------------------------------------------
# creat new environment and install GPU on it:
conda create -n tf
conda activete tf
conda install tensorflow-gpu
##---------------------------------------------------------------------