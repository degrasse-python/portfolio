from __future__ import print_function
import numpy as np
import math
import tensorflow as tf
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import position as pos

# restore_path = str(os.getenv('RESTORE_PATH', '/home/position_0.ckpt'))
saver_path = str(os.getenv('SAVER_PATH', '/home/postion/'))

# This example assumes a data source that provides an batch of entire flights. 
params={}
# Parameter dictionary
params = {}
params['points'] = 4 # 2 pairs of points 
params['true_sz'] = y_size = 2 # estimation of next pair of points
params['max_checkpoints'] = 4 # number of model checkpoint files to keep

tf.reset_default_graph()
model = pos.model(mode=pos.train_mode, params=params)# model.restore(restore_path)



for i in range(10001):
    start_t=time.time()  
    
    y_p, y_true_label, y_est_label, ce, lr = model.train( , )
            
    print("i", i, "time", time.time()-start_t, "reports ",steps,"steps src:", "{0:0.2f}%".format(),
          "ce: {0:0.2f}".format(), lr)
    if i%1==0:
        # now an example of a test run
        save_path = model.save(saver_path,i)
        
        
