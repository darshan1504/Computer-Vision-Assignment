# Importing all the libraries
import cv2
import pylab as plt
import numpy as np

# read the image
img = cv2.imread('image.jpg');
# convert to grayscale using open cv function
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

# Dividing into rows and columns using shape
image_rows = img.shape[0]
image_cols = img.shape[1]

empty_filter_matrix = []


# plotting original image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

# plotting original image
plt.subplot(2, 2, 3)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])



filter_size = input("Enter the size of filter 3 or 5 :")

if filter_size == 3:
    cv_median = cv2.medianBlur(img, 3)
    plt.subplot(2, 2, 4)
    plt.imshow(cv_median, cmap='gray')# Plots or shows the image and also sets the image to gray
    plt.title('Median Blur Image with FilterSize 3 Using OpenCV')
    plt.xticks([]), plt.yticks([]) # Get or set the x-limits and y-limits of the current tick locations and labels.
elif filter_size == 5:
    cv_median = cv2.medianBlur(img, 5)
    plt.subplot(2, 2, 4)
    plt.imshow(cv_median, cmap='gray')
    plt.title('Median Blur Image with FilterSize 5 Using OpenCV')
    plt.xticks([]), plt.yticks([])
else:
    print ("Invalid filter size. Please re-run the code")
    exit(0)

# Function to Calculate filter
def calculate_filter(i, j, n):
    if n == 3:
        # Creating the matrix of 3*3 and storing the neighbours values and if there are no neighbour values the storing zero as a border pixel
        for x in range(-1, 2):
            for y in range(-1, 2):
                try:
                    empty_filter_matrix.append(img[i + x][j + y]) # Appending the neighbour values in empty matrix
                except:
                    empty_filter_matrix.append(0)# Appending the 0 values in empty matrix
        return empty_filter_matrix
    if n == 5:
        # Creating the matrix of 5*5 and storing the neighbours values and if there are no neighbour values the storing zero as a border pixel
        for x in range(-2, 3):
            for y in range(-2, 3):
                try:
                    empty_filter_matrix.append(img[i + x][j + y]) # Appending the neighbour values in empty matrix
                except:
                    empty_filter_matrix.append(0) # Appending the 0 values in empty matrix
        return empty_filter_matrix


# For loop to slide filter over an image
for i in range(image_rows):
	for j in range(image_cols):
        # Calculating the filter using calculate_filter function
		filter = calculate_filter(i,j,filter_size)
        # Store the values in the relative pixel values
		img[i][j] = np.median(filter)
        # Delete the matrix after getting new values
		del filter[:]



# Plotting median filter image
plt.subplot(2, 2, 2)
plt.imshow(img, cmap='gray')
plt.title('Median Blur Image ')
plt.xticks([]), plt.yticks([])
# Show the plot
plt.show()
