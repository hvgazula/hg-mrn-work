
# coding: utf-8


#%%
# ### Imports
from sklearn.datasets import fetch_mldata
from sklearn import mixture
import pymc3 as pm
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

#%%
# ### Downloading the MNIST Data
custom_data_home = r'C:\Users\hgazula\Documents\Python Scripts'
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
X, y = mnist["data"], mnist["target"]

##%% Basic Printing
#print(X.shape)
#print(y.shape)
#print(np.unique(y))

#%%
# ### Divide data across local sites
# #### Part 01 ( Distribute fixed/known images at each local site)
# * Assume 3 local sites
# * Images at Local Site 01 (1, 3, 5)
# * Images at Local Site 02 (1, 2, 8)
# * Images at Local Site 03 (3, 5, 8)
#
# #### Part 02 ( Distribute images randomly at each local site)
# * To be done later once Part 01 is successful


#%%
# ### Assign image to each local site randomly
site1_images = np.random.randint(0, 10, size=(1, 3))
site2_images = np.random.randint(0, 10, size=(1, 3))
site3_images = np.random.randint(0, 10, size=(1, 3))

site1_images = [1, 3, 5]
site2_images = [1, 2, 8]
site3_images = [3, 5, 8]

#print(site1_images)
#print(site2_images)
#print(site3_images)

#%%
# ### Extract images to local site 01
site1_Xa = X[y == site1_images[0]]
site1_ya = y[y == site1_images[0]]
site1_Xb = X[y == site1_images[1]]
site1_yb = y[y == site1_images[1]]
site1_Xc = X[y == site1_images[2]]
site1_yc = y[y == site1_images[2]]

site1_X = np.vstack((site1_Xa, site1_Xb, site1_Xc))
site1_y = np.hstack((site1_ya, site1_yb, site1_yc))

#print(len(site1_X))
#print(len(site1_y))

#%%
# ### Extract images to local site 02
site2_Xa = X[y == site2_images[0]]
site2_ya = y[y == site2_images[0]]
site2_Xb = X[y == site2_images[1]]
site2_yb = y[y == site2_images[1]]
site2_Xc = X[y == site2_images[2]]
site2_yc = y[y == site2_images[2]]

site2_X = np.vstack((site2_Xa, site2_Xb, site2_Xc))
site2_y = np.hstack((site2_ya, site2_yb, site2_yc))

#print(len(site2_X))
#print(len(site2_y))

#%%
# ### Extract images to local site 02
site3_Xa = X[y == site3_images[0]]
site3_ya = y[y == site3_images[0]]
site3_Xb = X[y == site3_images[1]]
site3_yb = y[y == site3_images[1]]
site3_Xc = X[y == site3_images[2]]
site3_yc = y[y == site3_images[2]]

site3_X = np.vstack((site3_Xa, site3_Xb, site3_Xc))
site3_y = np.hstack((site3_ya, site3_yb, site3_yc))

#print(len(site3_X))
#print(len(site3_y))

#%% Time for density estimation (at each local site)
# fit a Gaussian Mixture Model with 3 components at each site
clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
clf.fit(site1_X)
mu1 = clf.means_
sigma1 = clf.covariances_
weights1 = clf.weights_

clf.fit(site2_X)
mu2 = clf.means_
sigma2 = clf.covariances_
weights2 = clf.weights_

clf.fit(site3_X)
mu3 = clf.means_
sigma3 = clf.covariances_
weights3 = clf.weights_

#%% Density aggregation at the remote site
w = np.hstack((weights1, weights2, weights3))
w = w/3
mu = np.vstack((mu1, mu2, mu3))
sigma = np.vstack((sigma1, sigma2, sigma3))

w1 = pm.floatX([.2, .8])
mu1 = pm.floatX([-.3, .5])
sd1 = pm.floatX([.1, .1])
with pm.Model() as model:
    obs = pm.NormalMixture('gmm', w1, mu1, sd1, dtype=theano.config.floatX)

    trace = pm.sample(10)

observed=np.random.randn(100)
#%% Resampling at each of the local sites (entails sending the distribution to the local sites)

#%% performing tSNE at each of the local sites and send the data to the remote site