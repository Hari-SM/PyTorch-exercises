import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#image = mpimg.imread('data/udacity_sdc.png')
image = mpimg.imread('data/curved_lane.jpeg')

plt.imshow(image)
plt.show()

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image, cmap='gray')
plt.show()

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = sobel_x.transpose()

filtered_image_x = cv2.filter2D(gray_image, -1, sobel_x)
filtered_image_y = cv2.filter2D(gray_image, -1, sobel_y)

fig, ax = plt.subplots(figsize=(16, 7), ncols=2)
ax[0].imshow(filtered_image_x, cmap='gray')
ax[0].set_title("Sobel_X filter")
ax[1].imshow(filtered_image_y, cmap='gray')
ax[1].set_title("Sobel_Y filter")
plt.show()
