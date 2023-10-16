import scipy.io
import numpy as np

infile='test_ivecs_filtered.mat'
outfile='test_ivecs_filtered.npz'

res = scipy.io.loadmat(infile)
print('loaded')

sp=[]
for f in res['Results']['wavename'][0,0][0]:
    sp.append(f[0])

np.savez_compressed(outfile,
                        data_path=sp,
                        features=np.transpose(res['Results']['ivectors'][0,0])
                        )