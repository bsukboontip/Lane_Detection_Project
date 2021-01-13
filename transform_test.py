#!/usr/bin/env python
# coding: utf-8

# In[25]:


import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from top_view_transform import *


# In[37]:


# Manual version takes a long time
# Read image from JPEG file
# Convert to grayscale
# pil_im = Image.open("6.jpeg")
pil_im = Image.open("sample6.jpeg")

width, height = pil_im.size
left = 0
top = height * 11 / 20
right = width
bottom = height - 220

pil_im = pil_im.crop((left, top, right, bottom)) 

orig = np.array(pil_im)
# plt.imshow(orig)
# plt.show()

# Resize so that standard height is 512
width = int(pil_im.width * 512 / pil_im.height)
pil_im = pil_im.resize((width , 512))

# Convert to np array
cropped_img = np.array(pil_im)
plt.imshow(orig_img)
plt.axis('off')
plt.show()

#Convert to grayscale
cropped_img = rgb2gray1(cropped_img)

cropped_height, cropped_width = cropped_img.shape

# # Deletes the 1 and 2 indices from the rgb parameter so the array can be squeezed
# im = np.delete(im, [1, 2], 2)

plt.imshow(im, cmap='gray')
plt.axis('off')
plt.show()


# In[39]:


# metric_h = 1
# alpha = 20
# theta_h = 10
# theta_v = 20

metric_h = 1
alpha = 40
theta_h = 40
theta_v = 10

transformed_img = top_view_transform(im, metric_h, math.radians(alpha), math.radians(theta_h), math.radians(theta_v))
print(transformed_img.shape)

# transformed_img = np.resize(transformed_img, (cropped_img.shape))

plt.imshow(transformed_img, cmap='gray')
plt.axis('off')
plt.show()

