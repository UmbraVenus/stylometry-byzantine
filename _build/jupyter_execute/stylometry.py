#!/usr/bin/env python
# coding: utf-8

# # Journal of Stylometry of Byzantine Ivory Boxes
# ### Author: Sage Ren, Diliana Angelova
# ### Date: 2022-Feb-18th
# ### Used Dataset: Cluny, Louvre Box, Veroli Box

# ## 1. Importing Prerequisites, Data Cleaning, and Augmentation

# In[1]:


get_ipython().system('pip install -U albumentations')


# In[2]:


#prerequisites
from fastai.vision.all import *
from fastai.data.external import *
from PIL import Image
import glob
import albumentations as A
import numpy as np 


# In[3]:


p = "dataset"
veroli = "Veroli Box"
louvre = "Louvre Box"
cluny = "Cluny"
v = "dataset/veroli/"
l = "dataset/louvre/"
c = "dataset/cluny/"
SIZE = 480
batch_size = 6


# In[4]:


# Image Augmentation Functions
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(),
])


# In[5]:


# Testing out a image with method resize

# saving data into Veroli List
veroli_list = []
for filename in glob.glob(veroli+"/*.JPG"):
    img = Image.open(filename)
    resized = img.resize((SIZE,SIZE))
    resized = np.array(resized)
    transformed_1 = transform(image=resized)['image']
    transformed_2 = transform(image=resized)['image']
    transformed_3 = transform(image=resized)['image']
    veroli_list.append(resized)
    veroli_list.append(transformed_1)
    veroli_list.append(transformed_2)
    veroli_list.append(transformed_3)

# saving data into Louvre List
louvre_list = []
for filename in glob.glob(louvre+"/*.JPG"):
    img = Image.open(filename)
    resized = img.resize((SIZE,SIZE))
    resized = np.array(resized)
    transformed_1 = transform(image=resized)['image']
    transformed_2 = transform(image=resized)['image']
    transformed_3 = transform(image=resized)['image']
    louvre_list.append(resized)
    louvre_list.append(transformed_1)
    louvre_list.append(transformed_2)
    louvre_list.append(transformed_3)

# saving data into Cluny list
cluny_list = []
for filename in glob.glob(cluny+"/*.JPG"):
    img = Image.open(filename)
    resized = img.resize((SIZE,SIZE))
    resized = np.array(resized)
    transformed_1 = transform(image=resized)['image']
    transformed_2 = transform(image=resized)['image']
    transformed_3 = transform(image=resized)['image']
    cluny_list.append(resized)
    cluny_list.append(transformed_1)
    cluny_list.append(transformed_2)
    cluny_list.append(transformed_3)

len(veroli_list), len(louvre_list), len(cluny_list)


# In[ ]:


cluny_list[-2]


# In[ ]:


# saving images into dataset folders
i = 0
for x in veroli_list:
    im = Image.fromarray(x)
    im.save(v+str(i)+".png")
    i += 1

i = 0
for x in louvre_list:
    im = Image.fromarray(x)
    im.save(l+str(i)+".png")
    i += 1
    
i = 0
for x in cluny_list:
    im = Image.fromarray(x)
    im.save(c+str(i)+".png")
    i += 1


# ## 2. Stylometry

# In[44]:


dls = ImageDataLoaders.from_folder(p, valid_pct=0.2, bs = batch_size)
dls.valid_ds.items[:10]


# In[45]:


dls.train_ds.items[:10]


# In[46]:


len(dls.train_ds.items)


# In[47]:


len(dls.valid_ds.items)


# In[48]:


dls.show_batch()


# In[49]:


learn = cnn_learner(dls, resnet34, metrics=error_rate)


# In[50]:


# at this point, we probabaly need to clear out memory
import torch 
torch.cuda.empty_cache()

import gc
gc.collect()


# In[51]:


torch.cuda.get_device_name(0)


# In[52]:


torch.cuda.memory_allocated(0)


# In[53]:


# at this point, we probabaly need to clear out memory
import torch 
torch.cuda.empty_cache()

import gc
gc.collect()

learn.lr_find()


# In[54]:


learn.fine_tune(4,3e-3)


# In[55]:


learn.show_results()


# In[56]:


interp = Interpretation.from_learner(learn)


# In[57]:


interp.plot_top_losses(9, figsize=(15,10))


# In[58]:


learn.predict("Veroli Box/IMG_0450.jpg")


# In[60]:


learn.predict("Louvre Box/IMG_0086.JPG")


# In[61]:


learn.predict("Cluny/IMG_0282.JPG")


# In[62]:


learn.predict("Cluny/IMG_0263.JPG")


# In[ ]:




