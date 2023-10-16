
#load data-----------------------------------------
import numpy as np
filename='test_kermanshah_xvec.npz'
with np.load(filename , allow_pickle=True) as da:
    data_name = da['data_name']
    fea = da['features']
    
    
filename='xvectors.npz'
    
with np.load(filename , allow_pickle=True) as da:
    spk_name = da['spk_name']
    feat = da['features']
#-------------------------------------------