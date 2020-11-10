#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

plt.style.use('seaborn-notebook')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


x = np.linspace(-10,10,500)
y = np.linspace(-10,10,500)
X,Y = np.meshgrid(x,y)
pos = np.empty((500,500,2))
pos[:,:,0]=X;pos[:,:,1]=Y


# In[4]:


mean=[0,0];var = np.diag([4,12])


# In[5]:


Z = multivariate_normal.pdf(pos,mean=mean,cov=var)


# In[ ]:




