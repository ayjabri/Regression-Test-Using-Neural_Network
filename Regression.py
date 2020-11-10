#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import statsmodels.api as sm
import scipy as sp
import matplotlib.pyplot as plt
from numpy.linalg import inv
from math import inf

plt.style.use('dark_background')


# # Regressions
# ## 1.Ordinary Least Squares $w_{LS}$

# In[3]:


np.random.seed(133)

N = 500
b = [1.5,2]
e = np.random.randn(500)

X = np.random.choice(np.linspace(-3,3,1000),size=N,replace=False)
X = np.column_stack((np.ones(N),X))
y = X@b + e

plt.scatter(X[:,1],y,s=5)


# In[4]:


# Ordinary Least Sqares

w_ols = inv(X.T@X)@(X.T@y)
y_ols = X@w_ols
residuals = sum(y_ols-y)

print("Residual Error:{}".format(residuals))
plt.figure(figsize=(10,6))
plt.scatter(X[:,1],y,s=5)
plt.plot(X[:,1],y_ols,c='orange',label=r'$y=b_0 + b_1 X$')
plt.legend();


# ## 2.Ridge Regression $w_{RR}$

# In[5]:


# Ridge Rgression
lam = [20,50,100,1000,2000,10000]

def Ridge(X,y,lam):
    I = np.identity(len(b))
    return inv(lam*I + X.T@X)@(X.T @y)

plt.figure(figsize=(10,6))
plt.scatter(X[:,1],y,s=5)
plt.plot(X[:,1],y_ols,c='yellow',label=r'$OLS$')
for l in lam:
    y_rr = X@Ridge(X,y,l)
    plt.plot(X[:,1],y_rr,label=f'$\lambda = {l}$',linewidth=0.25)
plt.legend();


# In[6]:


# Rideg regression with Data preprocessing

X[:,1:]= (X[:,1:]-X[:,1:].mean(axis=0))/X[:,1:].std(axis=0)
y = y-y.mean()
y_ls = X@Ridge(X,y,0)

plt.figure(figsize=(10,6))
plt.scatter(X[:,1],y,s=5,label='With data regulization')
plt.plot(X[:,1],y_ls,c='yellow',label=r'$OLS$')
for l in lam:
    y_rr = X@Ridge(X,y,l)
    plt.plot(X[:,1],y_rr,label=f'$\lambda = {l}$',linewidth=0.25)
plt.legend();


# ### Non-linear data

# In[7]:


np.random.seed(144)

N = 300
b = [1.5,2,-2]
e = np.random.normal(loc=0,scale=2,size=N)

x = np.random.choice(np.linspace(0,5,1000),size=N,replace=False)
X = np.column_stack((np.ones(N),x,np.log(x**2))) 
y = X@b + e


# In[8]:


# we add ones to X and X^2 and continue as usual
X[:5]


# In[9]:



y_ols = X@(inv(X.T@X)@(X.T@y))

# Ridge Rgression
lam = [20,50,100,1000,2000,10000]
s = np.argsort(x)

plt.figure(figsize=(10,6))
plt.scatter(x,y,s=5)
plt.plot(x[s],y_ols[s],c='yellow',label=r'$OLS$')
for l in lam:
    y_rr = X@Ridge(X,y,l)
    plt.plot(x[np.argsort(x)],y_rr[np.argsort(x)],label=f'$\lambda = {l}$',linewidth=0.5)
plt.legend();


# In[10]:


m = sm.OLS(y,X[:,-2:]) # drop the ones column
m.fit().summary()


# ## 3.Maximum Likelihood $w_{ML}$
# 
# The asumption is that $y\sim (Xw|\sigma^2 I)$

# In[11]:


np.random.seed(110)

N = 200
b = [-2,1.75]
e = np.random.normal(size=N)
x = np.linspace(-3,3,N)
X = np.column_stack((np.ones(N),x))
y = X@b + e

# y appears to be normally distributed
fig =plt.figure(num=1)
ax1 = fig.add_subplot(121)
ax1.hist(y,bins=40,color='cornflowerblue',ec='b')
ax2 = fig.add_subplot(122)
ax2.scatter(x,y,s=1);


# In[12]:


# bs is a vecotr that contains the intercept b0 and 
# define the log-likelihood function (minimize the negative = maximize likelihood)
def log_lik(bs,y,X):
    if bs[-1] <0: return inf
    lik = sp.stats.norm.pdf(y,loc=X@bs[:-1],scale=bs[-1])
    if all(v==0 for v in lik):return inf
    log_lik = np.log(lik[np.nonzero(lik)])
    return -sum(log_lik)


# In[13]:


from scipy.optimize import minimize

optim = minimize(log_lik,x0=bs,args=(y,X))


# In[14]:


optim


# In[15]:


w_ml = optim.x[:-1]
w_rr = Ridge(X,y,20)
w_ls = Ridge(X,y,0)


# In[16]:


y_ml = X@w_ml
y_rr = X@w_rr
y_ls = X@w_ls

plt.figure(figsize=(12,8))
plt.scatter(x,y,s=0.75)
plt.plot(x,y_ml,label='ML')
plt.plot(x,y_rr,label='Ridge')
plt.plot(x,y_ls,label='OLS')
plt.plot(x,X@b,label='Original')
plt.legend();


# In[17]:


print(f'Original:\t\t{b}\nMaximum Likelihood:\t{w_ml}\nRidge Regression:\t{w_rr}\nLeast Squares:\t\t{w_ls}')


# # 4.Maximum Posterior $w_{MAP}$
# 
# We define a distribution on $w_{MAP}$ instead of just one value  
# 
# $\mu = (\lambda\sigma^2I+X^TX)^{-1}X^Ty$  
# $\Sigma = (\lambda I +\sigma^{-2}X^TX)^{-1}$

# In[18]:


def MAP_regression(X,y,lam,sigma2):
    I = np.identity(X.shape[1])
    mu = inv(lam*sigma2*I + X.T@X)@X.T@y
    S = inv(lam*I+1/sigma2 * X.T@X)
    return mu,S


# In[19]:


np.random.seed(99)

N = 200
X = np.random.randn(N,3)
b = [1.5,2.25,-3.1]
epsilon = np.random.normal(loc=0,scale=2,size=N)
y = X@b + epsilon

sigma2 =2
lam = 0.1


# In[20]:


mu, S = MAP_regression(X,y,2,0.1);mu,S


# To predict a value $y_0$ given $x_0$ we actually predict a distribution of possible values $\mu_0$ and $\sigma^2_0$ of each prediction:    
# $\mu_0 = x^T_0\mu$  
# $\sigma^2_0 = \sigma^2 + x^T_0\Sigma x_0$

# In[21]:


mu_0 = X[0].T@mu
sigma2_0 = sigma2 + X[0].T@S@X[0]


# In[22]:


sigma2_0,mu_0


# In[23]:


a = np.linspace(-4,12,50)
e = sp.stats.norm.pdf(a,loc=mu_0,scale=sigma2_0)

plt.plot(a,e)
plt.plot(np.ones(50)*mu_0,np.linspace(0,0.21,50),c='r',linestyle='dashed')
plt.xlabel(r'$y_0$')
plt.title("Probability Distribution of $y_0$ given $x_0, \mu & \Sigma$")
plt.ylim(0,0.21)


# # 5. Minimum $l_2$ Regression
# 
# We use it if the number of features d is bigger than the number of samples n
# 

# In[ ]:





# # 6. LASSO Regression:
# 
# The same as Ridge regression in term of regularizing regression by penalizing w.  
# In ridge we multiplied $\lambda.Slope^2$  
# in LASSO we multiply $\lambda |Slope|$  
# 
# Sklean uses alpah for $\lambda$

# In[24]:


from sklearn.linear_model import Lasso


lr = Lasso(alpha=0.1)
lr.fit(X,y)


# In[25]:


lr.score(X,y)


# In[ ]:




