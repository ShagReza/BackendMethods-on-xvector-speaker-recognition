import numpy as np
from sklearn.metrics import roc_curve as rc
import matplotlib.pyplot as plt

def cosinedis(v1,v2):
  dot_product = np.dot(v1, v2)
  norm_a = np.linalg.norm(v1)
  norm_b = np.linalg.norm(v2)
  return dot_product / (norm_a * norm_b)

def getstring(aa):
   st=''
   for aaa in aa:
     st = st + ('%.4f' %(aaa)) + '\t'
   return st.strip()

def read_scp(filename,seperator='\t'):
   with open(filename , 'r') as fid:
      data = [ x.strip().split(seperator)   for x in fid]
   return data

def compute_eer(fpr,tpr,thresholds):
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]  

def findFR(x , fpr, tpr):
    abs_diffs = np.abs(fpr*100 - x)
    
    minval = np.min(abs_diffs)
    
    aaa = np.where(abs_diffs == minval)
    
    min_index = aaa[0][len(aaa[0])-1]
    fr = 1 - tpr[min_index]
    return fr*100


def showplot(fpr,tpr,threshold,eer,th , test_type , test_model , test_similarity , fr,l,data_name,logfile='log.txt',limitplot=1):
   plt.title(  data_name + '_' + test_type + '_' +  test_model + '_' + test_similarity + '\nEER=%.2f' %(eer*100) + '%')
   plt.plot(threshold, fpr, 'b' , label='FP=FA')
   plt.plot(threshold, 1-tpr, 'r',label='1-TP=FN=FR')
   plt.legend(loc = 'upper center')
   if (limitplot==1):
     plt.xlim([0, 1])
     plt.ylim([0, 1])
   plt.xlabel('Threshold')
   #plt.show()
   plt.savefig(data_name + '_' + test_type + '_' +  test_model + '_' + test_similarity +'_DET.png')
   
   plt.figure()
   plt.title( data_name + '_' +  test_type + '_' +  test_model + '_' + test_similarity + '\nEER=%.2f' %(eer*100) + '%')
   plt.plot(l, fr, '*b' )
   plt.xlabel('FA')
   plt.ylabel('FR')
   #plt.show()
   plt.savefig(data_name + '_' + test_type + '_' +  test_model + '_' + test_similarity +'.png')
   fi = open(logfile , 'a')
   fi.write('%s\t%s\t%s\t%s\n'  %(data_name,test_type,test_model,test_similarity))
   fi.write('EER = %.2f\n'  %(eer*100))
   fi.write(getstring(l) + '\n')
   fi.write(getstring(fr) + '\n')
   fi.write('*********************************************\n')
   fi.close()
   
 
def test(data, dic):
  print(str(len(data)))  
  y=[]
  y_pred=[]
  for i in range(len(data)):
    y.append(int(data[i][2]))
    file1 = dic[data[i][0]]
    file2= dic[data[i][1]]  
    res = cosinedis(file1,file2)
    y_pred.append(res)
  fpr, tpr, threshold = rc(y, y_pred, pos_label=1)
  eer , th =  compute_eer(fpr,tpr,threshold)   
  return fpr,tpr,threshold,eer,th    
   

   
def load_data(scorefile):
       dic={}
       with np.load(scorefile , allow_pickle=True) as da:
         a = da['data_path']
         b = da['features']
       print(a)
       for i in range(len(a)):
          dic[a[i]] = b[i]
       return dic
   
def saveplot(dic , l , scp,data_name,test_type,test_model,test_similarity,logfile,seperator='\t', limitplot=1):
  data  = read_scp(scp , seperator)
  fpr,tpr,threshold,eer,th = test(data, dic)
  fr = [findFR(x , fpr, tpr)  for x in l]
  showplot(fpr,tpr,threshold,eer,th ,test_type , test_model , test_similarity , fr,l,data_name,logfile,limitplot)