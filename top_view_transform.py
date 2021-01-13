import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
img = front view image to be transformed
metric_h = height of camera in cm
alpha = tilt angle of camera
theta_h = horizontal view angle of camera in radians
theta_v = vertial view angle of camera in radians
'''
def top_view_transform(img, metric_h, alpha, theta_h, theta_v):

    # img_transform = np.zeros_like(img)
    # img_transform = np.zeros((2100, 6800))

    u = img.shape[0]
    v = img.shape[1]
    print(u, v)

    l_min = metric_h * math.tan(alpha)
    # print(theta_h)
    w_min = 2 * l_min * math.tan(theta_h/2)
    # print(w_min, l_min)
    k = v / w_min
    print("k:", k, "w_min:", w_min)

    # h is the height of camera in pixels
    h = metric_h * k
    # print('h:', h)

    gamma = theta_v * (u - 0) / u
    l_i = h * math.tan(alpha + gamma)
    l_0 = h * math.tan(alpha)
    max_x = int(l_i - l_0)

    beta = theta_h * (v - (v - 1)) / v
    max_y = int(l_i * math.tan(theta_h - beta))

    print(f'max_x: {max_x}, max_y: {max_y}')
    img_transform = np.zeros((max_x + 1, max_y + 1))

    count = 0

    for u_i in range(u):
        for v_i in range(v):
            gamma = theta_v * (u - u_i) / u
            l_i = h * math.tan(alpha + gamma)
            l_0 = h * math.tan(alpha)

            # print(gamma, l_i, l_0)

            beta = theta_h * (v - v_i) / v

            x_i = max_x - int(l_i - l_0)
            y_i = int(l_i * math.tan(theta_h - beta))
            # if y_i == 0:
                # print(l_i, math.tan(theta_h - beta))
            # if count % 100 == 0:
                # print(f'l_i: {l_i}m l_0: {l_0}, u_i: {u_i}, v_i: {v_i}, x_i: {x_i}, y_i: {y_i},  gamma: {gamma}')
            
            # print(f'l_i: {l_i}m l_0: {l_0}, u_i: {u_i}, v_i: {v_i}, x_i: {x_i}, y_i: {y_i},  gamma: {gamma}')

            count += 1

            img_transform[x_i][y_i] = img[u_i][v_i]


    return img_transform

def rgb2gray1(rgb):
    # Sets each pixel to a weighted value
    grayed = 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]
    return grayed

if __name__ == "__main__":
    # Manual version takes a long time
    # Read image from JPEG file
    # Convert to grayscale
    pil_im = Image.open("6.jpeg")
    orig = np.array(pil_im)
    # plt.imshow(orig)
    # plt.show()

    # Resize so that standard height is 512
    width = int(pil_im.width * 512 / pil_im.height)
    pil_im = pil_im.resize((width , 512))

    # Convert to np array
    orig_img = np.array(pil_im)
    plt.imshow(orig_img)
    plt.axis('off')
    plt.show()

    #Convert to grayscale
    im = rgb2gray1(orig_img)

    # # Deletes the 1 and 2 indices from the rgb parameter so the array can be squeezed
    # im = np.delete(im, [1, 2], 2)

    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.show()

    metric_h = 1
    alpha = 70
    theta_h = 20
    theta_v = 10
    transformed_img = top_view_transform(im, metric_h, math.radians(alpha), math.radians(theta_h), math.radians(theta_v))

    plt.imshow(transformed_img, cmap='gray')
    plt.axis('off')
    plt.show()