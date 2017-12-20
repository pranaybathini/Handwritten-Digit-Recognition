
# coding: utf-8

# In[20]:

import gzip
import pickle

with gzip.open('mnist.pkl.gz') as f:
    train,val,test = pickle.load(f,encoding = 'latin1')
    x,y = train


# In[21]:

import matplotlib.cm as cm
import matplotlib.pyplot as plt

for y in range(20):
    plt.imshow(x[y].reshape((28,28)) ,cmap = cm.Greys_r)
    plt.show()

