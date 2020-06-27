import numpy as np
from os import listdir

if __name__ == '__main__':

  max_acc = 0.0
  max_acc_f = ''
   
  min_mse = 1e6
  min_mse_f = ''


  DIRECTORY='./experiment2/'
  for f in listdir(DIRECTORY):
    
    if 'depth1' not in f:
      continue
  
    if '_acc' in f:

      a = np.load(DIRECTORY+f)
      if a > max_acc:
        max_acc = a
        max_acc_f = f

    if '_loss' in f:
      
      mse = np.load(DIRECTORY+f)
      if mse < min_mse:
        min_mse = mse
        min_mse_f = f

  
  print('Best Acc:')
  print(max_acc)
  print(max_acc_f)
    

  print('MIN MSE:')
  print(min_mse)
  print(min_mse_f)
 
