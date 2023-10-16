from utils import saveplot,load_data

scorefile = 'test_ivecs_filtered.npz' 

l = [0 , 0.0001 , .001 , .01 , 0.1 , 0.2,0.3,0.4,0.5,0.75,1]
 
dic=load_data(scorefile)


saveplot(dic , l , 'test-all.scp','Kermanshah','ALL' , 'Ivector_raw' , 'Cosine' , 'log.txt','\t',1)





