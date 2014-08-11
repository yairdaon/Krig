'''
Created on Aug 1, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import config as cfg
import numpy as np
c = cfg.Config()
import numpy as np

a = [ -22,  347, 4448,  294,  835, 4439,  587,  326]
a = np.asarray(a)
a[np.where(a <= 0.0)] = -np.inf