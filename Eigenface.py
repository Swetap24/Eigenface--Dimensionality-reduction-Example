#!/usr/bin/env python
# coding: utf-8

# Dimensionality Reduction Example

# In[ ]:


from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from numpy import mean
from numpy import cov
from numpy.linalg import eig


# In[ ]:


def pca_fun(input_data, target_d):
    # P: d x target_d matrix containing target_d eigenvectors
    cov_mat=np.cov(input_data.T)
    Values, Vectors=eig(cov_mat)
    P=(Vectors[:,:target_d])
    return P


# In[ ]:


# Dataset used was face images from the Yale Face Database B, which contains face images from 10 people under 64 lighting conditions

data = loadmat('dataset')# (Dataset used was .mat file)
image = data['image'][0]
person_id = data['personID'][0]


# In[ ]:


def normalize_image(image):
  new_arr=[]
  for i in range (image.shape[0]):
    new_arr.append( np.reshape(image[i], (2500,1)))
  return new_arr


# In[ ]:


norm_img=np.array(normalize_image(image))


# In[ ]:


new=norm_img.reshape(640,2500)


# In[ ]:


m= pca_fun(new, 200)


# In[ ]:


new_m=m.T


# In[152]:


new_m[0].shape


# In[153]:


plt.imshow(np.real(new_m[1].reshape(50,50)), cmap='gray')
plt.show()


# In[155]:


plt.imshow(np.real(new_m[0].reshape(50,50)), cmap='gray')
plt.show()


# In[158]:


plt.imshow(np.real(new_m[3].reshape(50,50)), cmap='gray')
plt.show()


# In[159]:


plt.imshow(np.real(new_m[2].reshape(50,50)), cmap='gray')
plt.show()


# In[160]:



plt.imshow(np.real(new_m[4].reshape(50,50)), cmap='gray')
plt.show()

