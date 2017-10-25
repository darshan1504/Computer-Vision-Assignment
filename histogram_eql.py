# Importing all the libraries
import cv2
import pylab as plt
import numpy as np


# read the image
img = cv2.imread('image.jpg');
# convert to grayscale using open cv function
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

# Normalising the histogram image
def normalise_hist(histogram_image):
    col, row = histogram_image.shape # Adding shape value i.e rows and columns to variables
    normalised_array = [0.0] * 256 # Array of 256 with values 0
    for i in range(col):
        for j in range(row):
            normalised_array[histogram_image[i, j]] += 1
    return np.array(normalised_array) / (col * row)

# Cumalative function
def cumulative_sum(h):
    # finds cumulative sum of a numpy array, list
    return [sum(h[:i + 1]) for i in range(len(h))]


def histogram_equi(histogram_image):
    # calculate Histogram
    normalised_image = normalise_hist(histogram_image) # Get the normalise form of an image
    cumulative_sum_var = np.array(cumulative_sum(normalised_image))  # Cumulative distribution function
    convert_image = np.uint8(255 * cumulative_sum_var)  # Finding transfer function values a
    hist_col, hist_row = histogram_image.shape
    final_image = np.zeros_like(histogram_image) # Makes the array zero
    # Applying transfered values for each pixels
    for i in range(0, hist_col):
        for j in range(0, hist_row):
            final_image[i, j] = convert_image[histogram_image[i, j]]
    return final_image, convert_image # Return transformed image, original and new histogram and transform function

histogram_equalised_image, output = histogram_equi(gray_image) # Calling the histogram equilization function

# Performing histogram quilization using opencv
open_cv_image = cv2.equalizeHist(gray_image)

# Plot original image
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray') # Plots or shows the image and also sets the image to gray
plt.title('Original Image')
plt.xticks([]), plt.yticks([]) # Get or set the x-limits and y-limits of the current tick locations and labels.
# Plot histogram image
plt.subplot(2, 2, 2)
plt.imshow(histogram_equalised_image, cmap='gray')
plt.title('Histogram Equilised Image')
plt.xticks([]), plt.yticks([])
# Plot original image
plt.subplot(2, 2, 3)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
# Plot open cv image
plt.subplot(2, 2, 4)
plt.imshow(open_cv_image, cmap='gray')
plt.title('Histogram Equilised Image using OpenCV')
plt.xticks([]), plt.yticks([])
plt.show() # Show the plot

