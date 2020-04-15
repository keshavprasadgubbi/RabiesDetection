#!/usr/bin/env python
# coding: utf-8

# In[4]:


import image_slicer
#image_slicer.slice('kp.jpg', 9)


# In[6]:


from image_slicer import join
tiles = image_slicer.slice('kp.jpg', 9000, save=True)


# In[7]:


image = join(tiles)
image.save('kp-join.jpg')


# In[ ]:




