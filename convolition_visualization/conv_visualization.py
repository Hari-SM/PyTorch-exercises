import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from network import Network

## Read the input image
bgr_image = cv2.imread('data/udacity_sdc.png')

## Convert the image to grayscale
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

## Normalize the grayscale image, rescale the values to [0, 1]
gray_image = gray_image.astype("float32")/255

## Plot grayscale image
plt.imshow(gray_image, cmap='gray')
plt.show()

## Defining four differnt filters
filter_vals = np.array([[-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1]])
filters = np.array([filter_vals, 
                    -1*filter_vals, 
                    filter_vals.T, 
                    -1*filter_vals.T])

## Visualize all four filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title("Filter {}".format(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')

plt.show()

## Initialize the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Network(weight)

## Get all layers
gray_image_tensor = torch.from_numpy(gray_image).unsqueeze(0).unsqueeze(0)
conv_layer, activated_layer, pooled_layer = model(gray_image_tensor)

## Visualize all layers
fig = plt.figure(figsize=(20, 32))
for i in range(4):
    ax1 = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax2 = fig.add_subplot(4, 4, i+5, xticks=[], yticks=[])
    ax3 = fig.add_subplot(4, 4, i+9, xticks=[], yticks=[])
    ax4 = fig.add_subplot(4, 4, i+13, xticks=[], yticks=[])
    ax1.imshow(filters[i], cmap='gray')
    ax1.set_title("Filter {}".format(i+1))
    ax2.imshow(np.squeeze(conv_layer[0, i].data.numpy()), cmap='gray')
    ax2.set_title("Convolutional Layer {}".format(i+1))
    ax3.imshow(np.squeeze(activated_layer[0, i].data.numpy()), cmap='gray')
    ax3.set_title("Activated Layer {}".format(i+1))
    ax4.imshow(np.squeeze(pooled_layer[0, i].data.numpy()), cmap='gray')
    ax4.set_title("Pooled Layer {}".format(i+1))

# plt.tight_layout()
plt.show()
