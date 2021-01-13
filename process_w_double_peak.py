# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cv2 as cv
import numpy as np
from PIL import Image
#from Visualization import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

import os
#from ggplot import *
import seaborn as sns
import pandas as pd
import random as r

# %% [markdown]
# # Preprocessing

# %%
def rgb2gray(rgb):
    # Sets each pixel to a weighted value
    grayed = 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]
    return grayed


# %%
# Mask is a zero matrix
# fillMask fills an empty numpy array with 255 for pixels that fits inside the defined triangle
def fillMask(mask):
    h, w = mask.shape
    print(h)
    print(w)
    bottom_left = (h, 0)
    middle = (int(h/2), int(w/2))
    bottom_right = (h, w)
    
    for x, row in enumerate(mask):
        for y, col in enumerate(row):
            # Applying equations to left_bound and right_bound
            left_bound = (h - x) * middle[1] / middle[0]
            right_bound = x * middle[1] / middle[0]
            if y > left_bound and y < right_bound and x <= 400:
                mask[x][y] = 255
                
    return mask


# %%
def TempMask(mask):
    h, w = mask.shape
    h = h + 90
    print(h)
    print(w)
    middle = (int(h/2), int(w/2))
    print(middle)
    for x, row in enumerate(mask):
        for y, col in enumerate(row):
            # Applying equations to left_bound and right_bound
            left_bound = ((h - x) * middle[1] / middle[0]) - 70
            right_bound = (x * middle[1] / middle[0]) + 100
            if y > left_bound and y < right_bound and x <= 450:
                mask[x][y] = 255
                
    return mask


# %%
def squareMask(mask):
    h, w = mask.shape
    middle = (int(h/2), int(w/2))
    for x in range(h):
        for y in range(w):
            left_bound = 420
            right_bound = 500
            if y > left_bound and y < right_bound:
                mask[x][y] = 0
            else:
                mask[x][y] = 255
                
    return mask


# %%
def reverse_apply_mask(image, mask):
    for x, row in enumerate(mask):
        for y, col in enumerate(row):
            if mask[x][y] == 255:
                image[x][y] = 0
    return image


# %%
# For each non-zero pixel in mask, the corresponding pixel on image is kept (the rest of the pixels in mask is discarded)
def apply_mask(image, mask):
    for x, row in enumerate(mask):
        for y, col in enumerate(row):
            if mask[x][y] != 255:
                image[x][y] = 0
    return image


# %%
# Manual version takes a long time
# Read image from JPEG file
# Convert to grayscale
pil_im = Image.open("sample6.jpeg")
orig = np.array(pil_im)
# plt.imshow(orig)
# plt.show()

# Resize so that standard height is 512
width = int(pil_im.width * 512 / pil_im.height)
pil_im = pil_im.resize((width , 512))

# Convert to np array
orig_img = np.array(pil_im)
plt.imshow(orig_img)
#plt.axis('off')
plt.show()

#Convert to grayscale
im = rgb2gray(orig_img)

# # Deletes the 1 and 2 indices from the rgb parameter so the array can be squeezed
# im = np.delete(im, [1, 2], 2)

plt.imshow(im, cmap='gray')
#plt.axis('off')
plt.show()

# %% [markdown]
# # Sobel Edge Detection

# %%
#Implementing image smoothing
rows = im.shape[0]
cols = im.shape[1]

#input any kernel size
kernel_shape_row = 5
kernel_shape_col = 5

kernel = np.ones((kernel_shape_row, kernel_shape_col))/(kernel_shape_row * kernel_shape_col)
mag_smooth = np.zeros((rows,cols))
print(kernel.shape)

#pad with zeros on the border for full blurring of image
padded_gray = np.zeros((rows + kernel_shape_row - 1, cols + kernel_shape_col - 1))
index_1_row = (int) ((kernel_shape_row - 1)/2)
index_last_row = (int) (rows + ((kernel_shape_row - 1)/2))
index_1_col = (int) ((kernel_shape_col - 1)/2)
index_last_col = (int) (cols + ((kernel_shape_col - 1)/2))
padded_gray[index_1_row:index_last_row, index_1_col:index_last_col] = im
print(padded_gray.shape)

for x in range(rows):
    for y in range(cols):
        mag_smooth[x][y] = (kernel * padded_gray[x:x+kernel_shape_row, y:y+kernel_shape_col]).sum()     
        
print(mag_smooth.shape)
plt.imshow(mag_smooth, cmap='gray')
plt.axis('off')
plt.show()


# %%
##Implementing sobel edge detector
Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

mag_G = np.zeros((rows,cols))
#mag_G_thresh = np.zeros((rows,cols))
print(mag_G.shape)

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        sumx = (Gx * mag_smooth[i-1:i+2, j-1:j+2]).sum()
        sumy = (Gy * mag_smooth[i-1:i+2, j-1:j+2]).sum()
        mag_G[i][j] = np.sqrt(sumx**2 + sumy**2)
        
print('finished making new image')
print(mag_G.shape)
plt.imshow(mag_G, cmap='gray')
plt.axis('off')
plt.show()

# %% [markdown]
# # Cropping

# %%
# Own implementation of cropped image

plt.imshow(mag_G, cmap='gray')
plt.axis('off')
plt.show()
mask = np.zeros_like(mag_G)
mask = fillMask(mask)
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()
cropped_im = apply_mask(mag_G, mask)
plt.imshow(cropped_im, cmap='gray')
plt.axis('off')
plt.show()


# %%

mask2= np.zeros_like(mag_G)
mask2 = TempMask(mask2)
plt.imshow(mask2, cmap='gray')
#plt.axis('off')
plt.show()
cropped_im = apply_mask(cropped_im, mask2)
plt.imshow(cropped_im, cmap='gray')
plt.axis('off')
plt.show()


# %%
#Cuts unnecessary edges like tunnels, center lane markings etc.

mask3= np.zeros_like(mag_G)
mask3 = squareMask(mask3)
plt.imshow(mask3, cmap='gray')
#plt.axis('off')
plt.show()
#cropped_im = reverse_apply_mask(cropped_im, mask3)
cropped_im = apply_mask(cropped_im, mask3)
plt.imshow(cropped_im, cmap='gray')
plt.axis('off')
plt.show()

# %% [markdown]
# # Otsu's method

# %%
#implementation of Otsu's method
def otsu_thresh(grayed_image):
    #turn image into 1D array and get the histgram and number of bins
    hist, bin_edges = np.histogram(grayed_image.ravel(), 256)
    
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    #calculate the weights(probabilities) of each class
    wf = np.cumsum(hist)
    wb = np.cumsum(hist[::-1])[::-1]
    
    #calculate the mean of each class
    mu_f = np.cumsum(hist * bins) / wf
    mu_b = (np.cumsum((hist * bins)[::-1]) / wb[::-1])[::-1]
    
    #calculate the inter-class variance
    variance = wf[:-1] * wb[1:] * (mu_f[:-1] - mu_b[1:]) ** 2
    
    #find the index that maximizes the inter-class variance to get the threshold value
    max_idx = np.argmax(variance)
    threshold = bins[:-1][max_idx]
    #threshold = threshold * 2
    return threshold


# %%
#threshold on the entire image gives a lower threshold value than what we want
threshold = otsu_thresh(cropped_im)
print(threshold)


# %%
#Apply Otsu's to only masked area
cropped_part = []
for x, row in enumerate(mask):
        for y, col in enumerate(row):
            if mask[x][y] == 255:
                cropped_part.append(cropped_im[x][y])
cropped_part = np.array(cropped_part)
threshold = otsu_thresh(cropped_part)
print(threshold)
thresh_im = np.zeros_like(cropped_im)
for x, row in enumerate(cropped_im):
    for y, col in enumerate(row):
        if cropped_im[x][y] >= threshold:
            thresh_im[x][y] = 255
        else:
            thresh_im[x][y] = 0
            
plt.imshow(thresh_im, cmap="gray")
plt.axis('off')
plt.show()

hist = plt.hist(cropped_part, np.arange(0,256))
plt.show()


# %%
##Making histogram of threshold

maxValue = int(np.amax(cropped_im)) + 1
histogram, bin_edges = np.histogram(cropped_im, bins=maxValue, range=(0,maxValue))
# configure and draw the histogram figure
plt.figure()
plt.title("Histogram of Magnitude of Gradient")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0, maxValue])
plt.ylim([0, 50]) # must be changed manually after generating once for better visual analysis
plt.vlines(threshold, 0, 200, colors = 'r')

plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()

# %% [markdown]
# # Hough Transform

# %%
# Rohit's code
def Hough_lines(img, line_length):

    height, width = img.shape[:2]
    
    # Creates the accumulator space
    accumulator = np.zeros([180, int(math.sqrt(height ** 2 + width ** 2))])

    lines = np.array([[0, 0], [0, 0]])

    #line_length = 40

    # look for every pixel
    for y in range(0, height):
        for x in range(0, width):
            # if pixel is white (possible part of a line)
            if img[y][x] > 250:
                # try all angles
                # Step = 3 to decrease number of lines
                for theta in range(0, 180, 1):
                    p = int(x * math.cos(math.radians(theta)) + y * math.sin(math.radians(theta)))
                    accumulator[theta][p] += 1
                    # Check if it looks like line and if it's not in a list
                    if (accumulator[theta][p] > line_length) and (p not in lines[:, 0]) and (theta not in  lines[:, 1]):
                        # lines is an array of [rho, theta] pairs that has a lot of points in the accumulator space
                        lines = np.vstack((lines, np.array([p, theta])))

    # clean two first zeros
    lines = np.delete(lines, [0, 1], axis=0)
    
#     print(lines)

    return accumulator, lines


# %%
hough_space, point_stack = Hough_lines(thresh_im, 50)


# %%
plt.figure(figsize = (20,2))
plt.imshow(hough_space, cmap='gray')
plt.xlabel('rho')
plt.ylabel('theta')
plt.show()


# %%
# Outputs 2 different points on the line to draw the line
def hough_intersect (rho, theta, image):
    h, w = image.shape[:2]
    out = []
    line_info = []
    theta = math.radians(theta)
    intersect = [int(round(rho / math.sin(theta))), int(round((rho - w * math.cos(theta)) / math.sin(theta))), int(round(rho / math.cos(theta))),
                 int(round((rho - h * math.sin(theta)) / math.cos(theta)))]
    
    
    # Use formula to get 2 points in a line, one being 100 pixels from left and another being 100 pixels from right
    x1 = 100
    y1 = int(x1 * (-math.cos(theta)/math.sin(theta)) + rho / math.sin(theta))
    
    x2 = 800
    y2 = int(x2 * (-math.cos(theta)/math.sin(theta)) + rho / math.sin(theta))
    out.append((x1, y1))
    out.append((x2, y2))
    # Gradient
    m = -math.cos(theta)/math.sin(theta)
    #print(m)
    # Intersect
    b = rho / math.sin(theta)
    line_info.append(m)
    line_info.append(b)
    
    return out, line_info


# %%
#test = hough_intersect(443, 69, orig)


# %%
points_coord = [] #To store the coordinates of the lines
orig_line = orig_img.copy()
edge_line = cropped_im.copy()

for i in point_stack:
    points, line_info = hough_intersect(i[0], i[1], orig)
    if (len(points) > 1 and ((line_info[0] < -0.5 and line_info[0] > -0.75) or (line_info[0] > 0.5 and line_info[0] < 0.75))):
        points_coord.append([list(point) for point in points])
        cv.line(orig_line, points[0], points[1], (255,0,0))
        cv.line(edge_line, points[0], points[1], (255,255,255))

fig = plt.imshow(orig_line)
#plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

plt.show()
plt.imshow(edge_line, cmap="gray")
#plt.axis('off')
plt.show()


# %%

df = pd.DataFrame({'rho':point_stack[:,0], 'theta':point_stack[:,1]})
bin_size = 13
sns.jointplot(data=df,x='rho',y='theta',marginal_kws=dict(bins=bin_size))


# %%
hist_rho, edges_rho = np.histogram(df['rho'].tolist(), bins=10)
cumsum_rho = np.cumsum(hist_rho)
hist_theta, edges_theta = np.histogram(df['theta'].tolist(), bins=10)
cumsum_theta = np.cumsum(hist_theta)


# %%
def groupContious(arr):
    arr.sort()
    i = 0
    rtn_arr = []
    while i < len(arr):
        start_i = arr[i]
        end_i = arr[i]
        while i < len(arr)-1 and arr[i] == arr[i+1]-1:
            end_i = arr[i+1]
            i += 1
        rtn_arr.append((start_i,end_i))
        i += 1
    return rtn_arr


# %%
rho_peeks = []
theta_peeks = []

if hist_rho[0] != 0 and hist_rho[0] >= hist_rho[1]:
    rho_peeks.append(0)

if hist_rho[-1] != 0 and hist_rho[-1] >= hist_rho[-2]:
    rho_peeks.append(len(hist_rho)-1)

if hist_theta[0] != 0 and hist_theta[0] >= hist_theta[1]:
    theta_peeks.append(0)

if hist_theta[-1] != 0 and hist_theta[-1] >= hist_theta[-2]:
    theta_peeks.append(len(hist_theta)-1)

for i in range(1, len(hist_rho)-1):
    if hist_rho[i] != 0 and hist_rho[i] >= hist_rho[i-1] and hist_rho[i] >= hist_rho[i+1]:
        rho_peeks.append(i)
    if hist_theta[i] != 0 and hist_theta[i] >= hist_theta[i-1] and hist_theta[i] >= hist_theta[i+1]:
        theta_peeks.append(i)
        
rho_peeks = groupContious(rho_peeks)
theta_peeks = groupContious(theta_peeks)


# %%
cleaned_point_stack = []

for rho_split in rho_peeks:
    rho_min, rho_max = edges_rho[rho_split[0]], edges_rho[rho_split[1]+1]
#     print(f'Between rho: [{rho_min}, {rho_max}]')
    matched_rho = df[df['rho'].between(rho_min, rho_max)]
    
    for theta_split in theta_peeks:
        theta_min, theta_max = edges_theta[theta_split[0]], edges_theta[theta_split[1]+1]
#         print(f'Between theta: [{theta_split[0]}, {theta_split[1]+1}]')
        
        matched_rho_theta =  df[df['theta'].between(theta_min, theta_max)]
        
        if matched_rho_theta.size != 0:
            sub_df_mean = matched_rho_theta.mean()
            cleaned_point_stack.append([sub_df_mean['rho'], sub_df_mean['theta']])


# %%
#Original Louis code for output comparison
points_coord = [] #To store the coordinates of the lines
orig_line = orig_img.copy()
edge_line = cropped_im.copy()

for i in cleaned_point_stack:
    points, lineinfo = hough_intersect(i[0], i[1], orig)
#     print(points)
    if (len(points) > 1):
        points_coord.append([list(point) for point in points])
        cv.line(orig_line, points[0], points[1], (255,0,0),2)
        cv.line(edge_line, points[0], points[1], (255,255,255),2)

fig = plt.imshow(orig_line)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

plt.show()
plt.imshow(edge_line, cmap="gray")
plt.axis('off')
plt.show()


# %%
#Only store lines averages that are closest to the actual lane lines
orig_line = orig_img.copy()
edge_line = cropped_im.copy()

pos_count = 0
neg_count = 0
pos_lines_b = []
pos_lines_m = []
neg_lines_b = []
neg_lines_m = []
points = []
line_info = []

j = 0
for i in cleaned_point_stack:
    point, lineinfo = hough_intersect(i[0], i[1], orig)
    if(len(point) > 1 and j < (len(cleaned_point_stack)/3)):
        print(lineinfo)
        if(lineinfo[0] > 0.3):
            pos_count += 1
            pos_lines_m.append(lineinfo[0])
            pos_lines_b.append(lineinfo[1])
        elif(lineinfo[0] < -0.3):
            neg_count += 1
            neg_lines_m.append(lineinfo[0])
            neg_lines_b.append(lineinfo[1])
    j += 1

print(pos_count)
print(neg_count)

min_pos_line_slope = min(pos_lines_m)
print(min_pos_line_slope)

max_neg_line_slope = max(neg_lines_m)
print(max_neg_line_slope)

#get rid of redundant lines because sometimes it overcounts
k = 1
for i in neg_lines_m:
    if(k < len(neg_lines_m)):
        if(i == (neg_lines_m[k]) and neg_count > 1):
            neg_count -= 1
        k += 1
            
k = 1
for i in pos_lines_m:
    if(k < len(pos_lines_m)):
        if(i == pos_lines_m[k] and pos_count > 1):
            pos_count -= 1
        k += 1

print(pos_count)
print(neg_count)
        
#add only two lines to the points set with the line information for each of those lines
j = 0
for i in cleaned_point_stack:
    point, lineinfo = hough_intersect(i[0], i[1], orig)
    print(lineinfo)
    if(len(point) > 1 and j < (len(cleaned_point_stack)/3)):
        if(pos_count > 1 and neg_count > 1 and lineinfo[0] > min_pos_line_slope and lineinfo[0] < max_neg_line_slope):
            print('1')
            if(lineinfo[0] > 0.3 or lineinfo[0] < -0.3):
                points.append(point)
                line_info.append(lineinfo)
        if(pos_count > 1 and neg_count == 1):
            print('2')
            if((lineinfo[0] > 0.3 and lineinfo[0] > min_pos_line_slope) or lineinfo[0] < -0.3):
                points.append(point)
                line_info.append(lineinfo)
        elif(pos_count == 1 and neg_count > 1):
            print('3')
            if(lineinfo[0] > 0.3 or (lineinfo[0] < -0.3) and lineinfo[0] < max_neg_line_slope):
                points.append(point)
                line_info.append(lineinfo)
        elif(pos_count == 1 and neg_count == 1):
            print('4')
            if(lineinfo[0] > 0.3 or lineinfo[0] < -0.3):
                points.append(point)
                line_info.append(lineinfo)
            
    j += 1

print(points)
print(line_info)

#print out the lines
for point in points:
    #if(len(points) > 1):
    points_coord.append([list(point) for point in points])
    cv.line(orig_line, point[0], point[1], (255,0,0),2)
    cv.line(edge_line, point[0], point[1], (255,255,255),2)
        
for info in line_info:
    if(info[0] < 0):
        neg_m = info[0]
        neg_b = info[1]
    else:
        pos_m = info[0]
        pos_b = info[1]

fig = plt.imshow(orig_line)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

plt.show()
plt.imshow(edge_line, cmap="gray")
plt.axis('off')
plt.show()

plt.imsave('temp.png', orig_line)


# %%
# Compute Vanishing Point

A = np.array([[-neg_m, 1], [-pos_m, 1]])
B = np.array([neg_b, pos_b])
C = np.linalg.solve(A, B)
h, w = orig_img.shape[:2]
vanished_img = orig_img.copy()
y1 = pos_m * (w - 100) + pos_b
cv.line(vanished_img, (int(C[0]), int(C[1])), (w - 100, int(y1)), (255,0,0), 2)

y2 = neg_m * 100 + neg_b
cv.line(vanished_img, (100, int(y2)), (int(C[0]), int(C[1])), (255,0,0), 2)

fig = plt.imshow(vanished_img)
plt.axis('off')
plt.show()


# %%



