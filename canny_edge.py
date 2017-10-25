# Importing all the libraries
import cv2
import pylab as plt
import numpy as np


img = cv2.imread('image.jpg');
# convert to grayscale using open cv function
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

# Plot the original image in the plot area
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray') # Plots or shows the image and also sets the image to gray
plt.title('Original Image')
plt.xticks([]), plt.yticks([]) # Get or set the x-limits and y-limits of the current tick locations and labels.

# checking the rows and column length of the image and storing them in the variables
image_rows = img.shape[0]
image_cols = img.shape[1]

# Smoothing using OpenCV gaussing
gaussian_blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

# Calculating the derrivative using Sobel

# Gradient-x using OpenCV sobel
gradient_x = cv2.Sobel(gaussian_blur_image, cv2.CV_16S , 1, 0, ksize=3)

# Gradient-y using OpenCV sobel
gradient_y = cv2.Sobel(gaussian_blur_image, cv2.CV_16S , 0, 1, ksize=3)

# Absolute of gradient-x and gradient-y
abs_grad_x = cv2.convertScaleAbs(gradient_x)
abs_grad_y = cv2.convertScaleAbs(gradient_y)

# Calculate the weighted sums using OpenCV
dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Calculating the grad_dir for thr non-max suppression, merging both image derivatives (in both X and Y grad_dir) to find the grad_dir and final image
grad_dir = np.arctan2(gradient_y, gradient_x)

# Creating the gradient angle to sectors depending on the pixel values in the gradient direction Matrix
for x in range(image_rows):
    for y in range(image_cols):
        if (grad_dir[x][y] < 22.5 and grad_dir[x][y] >= 0) or \
                (grad_dir[x][y] >= 157.5 and grad_dir[x][y] < 202.5) or \
                (grad_dir[x][y] >= 337.5 and grad_dir[x][y] <= 360):
            grad_dir[x][y] = 0
        elif (grad_dir[x][y] >= 22.5 and grad_dir[x][y] < 67.5) or \
                (grad_dir[x][y] >= 202.5 and grad_dir[x][y] < 247.5):
            grad_dir[x][y] = 45
        elif (grad_dir[x][y] >= 67.5 and grad_dir[x][y] < 112.5) or \
                (grad_dir[x][y] >= 247.5 and grad_dir[x][y] < 292.5):
            grad_dir[x][y] = 90
        else:
            grad_dir[x][y] = 135

non_max_supression = dst.copy()
# calculation the non max suppression for each gradient direction angle
# checking for the pixels behind and ahead and set them to zero if selected pixel is small from neighbours
for x in range(1, image_rows - 1):
    for y in range(1, image_cols - 1):
        if grad_dir[x][y] == 0:
            if (dst[x][y] <= dst[x][y + 1]) or \
                    (dst[x][y] <= dst[x][y - 1]):
                non_max_supression[x][y] = 0
        elif grad_dir[x][y] == 45:
            if (dst[x][y] <= dst[x - 1][y + 1]) or \
                    (dst[x][y] <= dst[x + 1][y - 1]):
                non_max_supression[x][y] = 0
        elif grad_dir[x][y] == 90:
            if (dst[x][y] <= dst[x + 1][y]) or \
                    (dst[x][y] <= dst[x - 1][y]):
                non_max_supression[x][y] = 0
        else:
            if (dst[x][y] <= dst[x + 1][y + 1]) or \
                    (dst[x][y] <= dst[x - 1][y - 1]):
                non_max_supression[x][y] = 0

# applying the hysterisis threshold on the non-max suppressed image
# We have the suppressed image we have to take two threshold values
# Setting up the two threshold values
# Try changing the values of the high threshold and low threshold for different outputs
# usualy keeping the high and low threshold as  high_threshold = 2 low_threshold
high_threshold = 45
low_threshold = 25
# storing the pixel values which are higher then the high threshold they contribute to final edges
strong_edges = (non_max_supression > high_threshold)

# Strong has value 2, weak has value 1
thresholded_edges = np.array(strong_edges, dtype=np.uint8) + (non_max_supression > low_threshold)

# Tracing edges with hysteresis, Find weak edge pixels near strong edge pixels
final_edges = strong_edges.copy() # Creating copy of strong edges
new_pixels = []
for r in range(1, image_rows - 1):
    for c in range(1, image_cols - 1):
        if thresholded_edges[r, c] != 1:
            continue  # Not a weak pixel
             # If the gradient at a pixel connected to an edge pixel is between Low and High then declare it an edge pixel directly or via pixels between Low and High
            local_patch = thresholded_edges[r - 1:r + 2, c - 1:c + 2]
            patch_max = local_patch.max()
            if patch_max == 2:
                new_pixels.append((r, c))
                final_edges[r, c] = 1
                # Extend strong edges based on current pixels
while len(new_pixels) > 0:
    new_pix = []
    for r, c in new_pixels:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                    r2 = r + dr
                    c2 = c + dc
                if thresholded_edges[r2, c2] == 1 and final_edges[r2, c2] == 0:
                    # Copy this weak pixel to final result
                    new_pix.append((r2, c2))
                    final_edges[r2, c2] = 1
    new_pixels = new_pix

cv_canny_edges = cv2.Canny(img,100,200)
plt.subplot(2, 2, 2)
plt.imshow(final_edges, cmap='gray') # Plots or shows the image and also sets the image to gray
plt.title('Finale Edge Image')
plt.xticks([]), plt.yticks([]) # Get or set the x-limits and y-limits of the current tick locations and labels.

plt.subplot(2, 2, 3)
plt.imshow(gray_image, cmap='gray') # Plots or shows the image and also sets the image to gray
plt.title('Original Image')
plt.xticks([]), plt.yticks([]) # Get or set the x-limits and y-limits of the current tick locations and labels.

plt.subplot(2, 2, 4)
plt.imshow(cv_canny_edges, cmap='gray') # Plots or shows the image and also sets the image to gray
plt.title('Finale Edge Image using OpenCV')
plt.xticks([]), plt.yticks([]) # Get or set the x-limits and y-limits of the current tick locations and labels.

plt.show()