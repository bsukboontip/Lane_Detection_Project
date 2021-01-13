#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from Preprocessing import *
from EdgeDetection import *
from HoughTransform import *
from ColorDetection import *
from PostProcessing import *
from Visualization import *


# # Load Image

# In[2]:


pil_im = Image.open("8.jpeg")
org = np.array(pil_im)
orig_img = org.copy()


# In[3]:


plt.imshow(org)


# # Resize

# In[4]:


resized = resize(pil_im,512)
resized.shape, org.shape


# # Grayscale

# In[5]:


gray = rgb_to_gray(resized)
plt.imshow(gray,cmap='gray')


# # Smoothing

# In[6]:


smooth = smoothing(gray)
plt.imshow(smooth,cmap='gray')


# # Sobel Edge Detector

# In[7]:


edge = sobel_edge_detector(smooth)
plt.imshow(edge,cmap='gray')


# # Masking

# In[8]:


mask = fill_mask(edge)
plt.imshow(mask,cmap='gray')


# In[9]:


cropped = apply_mask(edge, mask)
plt.imshow(cropped,cmap='gray')


# In[10]:


mask = TempMask(cropped)
plt.imshow(mask,cmap='gray')


# In[11]:


cropped = apply_mask(edge, mask)
plt.imshow(cropped,cmap='gray')


# In[12]:


mask = squareMask(cropped)
plt.imshow(mask,cmap='gray')


# In[13]:


cropped = reverse_apply_mask(cropped, mask)
plt.imshow(cropped,cmap='gray')


# # Thresholding

# In[14]:


t = otsu_threshold(cropped)
t


# In[15]:


threshed = apply_threshold(cropped, t)
plt.imshow(threshed,cmap='gray')


# # Hough Transform

# In[16]:


hough_space, point_stack = hough_lines(threshed, 50)

orig_line, gradients, line_info = draw_point_stack(resized, point_stack)

plt.imshow(orig_line)
plt.axis('off')
plt.show()


# # Double Peak Averaging

# In[17]:


cleaned_point_stack = double_peak_grouping(point_stack)

orig_line, gradients, line_info = draw_point_stack(resized, cleaned_point_stack)

plt.imshow(orig_line)
plt.axis('off')
plt.show()


# In[18]:


# Exclude lines not from lanes
h, w = resized.shape[:2]

neg_lines = [line for line in line_info if line[0] < -0.3]
pos_lines = [line for line in line_info if line[0] > 0.3]

cleaned_lines_img = resized.copy()

for coeff in pos_lines:
    m = coeff[0]
    b = coeff[1]
    y1 = m * 100 + b
    y2 = m * (w - 100) + b
    cv.line(cleaned_lines_img, (100, int(y1)), (w - 100, int(y2)), (255,0,0))
    
for coeff in neg_lines:
    m = coeff[0]
    b = coeff[1]
    y1 = m * 100 + b
    y2 = m * (w - 100) + b
    cv.line(cleaned_lines_img, (100, int(y1)), (w - 100, int(y2)), (255,0,0))

fig = plt.imshow(cleaned_lines_img)
plt.show()


# In[19]:


# Compute Average Lines
averaged_lines_img = resized.copy()

total_m = 0
total_b = 0
for coeff in pos_lines:
    total_m += coeff[0]
    total_b += coeff[1]
    
pos_m = total_m / len(pos_lines)
pos_b = total_b / len(pos_lines)

total_m = 0
total_b = 0
for coeff in neg_lines:
    total_m += coeff[0]
    total_b += coeff[1]
    
neg_m = total_m / len(neg_lines)
neg_b = total_b / len(neg_lines)

y1 = pos_m * 100 + pos_b
y2 = pos_m * (w - 100) + pos_b
cv.line(averaged_lines_img, (100, int(y1)), (w - 100, int(y2)), (255,0,0))

y1 = neg_m * 100 + neg_b
y2 = neg_m * (w - 100) + neg_b
cv.line(averaged_lines_img, (100, int(y1)), (w - 100, int(y2)), (255,0,0))

fig = plt.imshow(averaged_lines_img)
plt.show()


# In[20]:


# Compute Vanishing Point
A = np.array([[-neg_m, 1], [-pos_m, 1]])
B = np.array([neg_b, pos_b])
C = np.linalg.solve(A, B)

vanished_img = resized.copy()


y1 = pos_m * (w - 100) + pos_b
cv.line(vanished_img, (int(C[0]), int(C[1])), (w - 100, int(y1)), (255,0,0))

y2 = neg_m * 100 + neg_b
cv.line(vanished_img, (100, int(y2)), (int(C[0]), int(C[1])), (255,0,0))

fig = plt.imshow(vanished_img)
plt.show()


# # Color Detection

# In[21]:


#add original image back
orig_img = np.array(pil_im)
plt.imshow(orig_img)
plt.title('Resized color image')
plt.axis('off')
plt.show()

colorMask = fill_color_mask(orig_img)
plt.imshow(colorMask)
plt.title('Color mask')
plt.axis('off')
plt.show

croppedImageColor = apply_mask_color(orig_img, colorMask)
plt.imshow(croppedImageColor)
plt.title('Color mask applied')
plt.axis('off')
plt.show()


# ## RBG to HSV

# In[22]:


hsv = rgb_to_hsv(croppedImageColor)


# In[23]:


#extract hue, saturation, and value
hueImg = hsv[:,:,0]
satImg = hsv[:,:,1]
valImg = hsv[:,:,2]

plt.imshow(croppedImageColor)
plt.title("RGB image (original resized with mask)")
plt.axis('off')
plt.show()

plt.imshow(hueImg, cmap = 'hsv')
plt.title("Hue channel")
plt.axis('off')
plt.show()

plt.imshow(satImg)
plt.title("Saturation channel")
plt.axis('off')
plt.show()

plt.imshow(valImg)
plt.title("Value channel")
plt.axis('off')
plt.show()


# ## Histogram hue, saturation, and value

# In[24]:


#use only the pixels that are both in the cropped section and on the lines
croppedHue = []
rows = colorMask.shape[0]
cols = colorMask.shape[1]
for y in range(rows):
    for x in range(cols):
        if (colorMask[y][x][0] != 0): #unmasked region
            yLine1 = int(neg_m * x + neg_b) #find lane line y value corresponding to the x value for the left lane line
            yLine2 = int(pos_m * x + pos_b) #find lane line y value corresponding to the x value for the right lane line
            if ((y == yLine1) or (y == yLine2)): #pixel on line
                    croppedHue.append(hueImg[y][x])
croppedHue = np.array(croppedHue)

maxHueValue = int(np.amax(croppedHue)) + 1

hueHistogram, hueBinEdges = np.histogram(croppedHue, bins=maxHueValue, range=(0,maxHueValue))
# configure and draw the histogram figure
plt.figure()
plt.title("Hue Histogram")
plt.xlabel("Hue (degrees)")
plt.ylabel("pixels")
plt.xlim([0, maxHueValue])
plt.ylim([0, 30]) # must be changed manually after generating once for better visual analysis

plt.plot(hueBinEdges[0:-1], hueHistogram)  # <- or here
plt.show()

#use only the cropped section
croppedSat = []
for y in range(rows):
    for x in range(cols):
        if (colorMask[y][x][0] != 0): #unmasked region
            yLine1 = int(neg_m * x + neg_b) #find lane line y value corresponding to the x value for the left lane line
            yLine2 = int(pos_m * x + pos_b) #find lane line y value corresponding to the x value for the right lane line
            if ((y == yLine1) or (y == yLine2)): #pixel on line
                croppedSat.append(satImg[y][x])
croppedSat = np.array(croppedSat)

maxSatValue = int(np.amax(croppedSat)) + 1

satHistogram, satBinEdges = np.histogram(croppedSat, bins=maxSatValue, range=(0,maxSatValue))
# configure and draw the histogram figure
plt.figure()
plt.title("Saturation Histogram")
plt.xlabel("Saturation (permille)")
plt.ylabel("pixels")
plt.xlim([0, maxSatValue])
plt.ylim([0, 10]) # must be changed manually after generating once for better visual analysis

plt.plot(satBinEdges[0:-1], satHistogram)  # <- or here
plt.show()

#use only the cropped section
croppedVal = []
for y in range(rows):
    for x in range(cols):
        if (colorMask[y][x][0] != 0): #unmasked region
            yLine1 = int(neg_m * x + neg_b) #find lane line y value corresponding to the x value for the left lane line
            yLine2 = int(pos_m * x + pos_b) #find lane line y value corresponding to the x value for the right lane line
            if ((y == yLine1) or (y == yLine2)): #pixel on line
                croppedVal.append(valImg[y][x])
croppedVal = np.array(croppedVal)

maxValValue = int(np.amax(croppedVal)) + 1

valHistogram, valBinEdges = np.histogram(croppedVal, bins=maxValValue, range=(0,maxValValue))
# configure and draw the histogram figure
plt.figure()
plt.title("Value Histogram")
plt.xlabel("Value (permille)")
plt.ylabel("pixels")
plt.xlim([0, maxValValue])
plt.ylim([0, 20]) # must be changed manually after generating once for better visual analysis

plt.plot(valBinEdges[0:-1], valHistogram)  # <- or here
plt.show()


# In[ ]:


# #Temporary thresholding for research purposes
# for y in range(rows):
#     for x in range(cols):
#         if valImg[y][x] < 800:
#             imHSV[y][x][2] = 0
#             valImg[y][x] = 0

# plt.imshow(imHSV)
# plt.show

# plt.imshow(valImg)
# plt.show


# ## Detection

# In[ ]:


#currently can only detect yellow and white since it treats the unmarked pavement as red which will be fixed later, needs to be tested with multiple images

#number of pixels with that color
yellows = 0
whites = 0
reds = 0

for y in range(rows):
    for x in range(cols):
        if (colorMask[y][x][0] != 0): #unmasked region
            yLine1 = int(neg_m * x + neg_b) #find lane line y value corresponding to the x value for the left lane line
            yLine2 = int(pos_m * x + pos_b) #find lane line y value corresponding to the x value for the right lane line
            if ((y == yLine1) or (y == yLine2)): #pixel on line
                pixelColor = get_color(hueImg[y][x], satImg[y][x], valImg[y][x]) #get the color (string) of that pixel
                if pixelColor == "yellow":
                    yellows += 1
                elif pixelColor == "white":
                    whites += 1
                #elif pixelColor == "red":
                    #reds += 1

#decide the most dominant color
if yellows == max(yellows, whites, reds):
    laneColor = "yellow"
elif reds == max(yellows, whites, reds):
    laneColor = "red"
elif whites == max(yellows, whites, reds):
    laneColor = "white"


#xTest = 330
#yTest = int(m1 * xTest + c1)
#print(f'{croppedImageColor[yTest][xTest][0]} {croppedImageColor[yTest][xTest][1]} {croppedImageColor[yTest][xTest][2]}')
print(f'There are {yellows} yellow pixels on the lane lines in the mask.')
print(f'There are {whites} white pixels on the lane lines in the mask.')
#print(f'There are {reds} red pixels on the lane lines in the mask.')
print(f'The color of the lane markings is {laneColor}.')

plt.imshow(croppedImageColor)
plt.axis('on')
plt.show


# In[ ]:





# In[ ]:




