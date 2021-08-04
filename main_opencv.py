# Import library
import os, cv2
import numpy as np
import matplotlib.pyplot as plt

# Gets current working directory
current_path = os.getcwd()
# Set input image name
image_name = "ArtGallery.jpg"
# Sets input image path
image_path = os.path.join(current_path, image_name)
# Read input image with cv2
image = cv2.imread(image_path)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# Creates a copy of original input image
spare_image = image.copy()

# Defines source points of first region
src_points = np.array([[38, 102], 
                       [341, 124], 
                       [342, 426], 
                       [42, 447]])

# Defines destination points of second region
dst_points = np.array([[683, 119], 
                       [956, 68], 
                       [955, 515], 
                       [678, 452]])

################ PART ONE - MOVE FIRST REGION TO ANOTHER REGION ################

# Finds a perspective transformation between two planes
homo_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
print(homo_matrix)

# Applies a perspective transformation to an image
persp_transform = cv2.warpPerspective(spare_image, homo_matrix, 
                        (spare_image.shape[1], spare_image.shape[0]))

# Draws a filled convex polygon with black color
spare_image = cv2.fillConvexPoly(spare_image, dst_points.astype(int), 0, 16)

# Fills the black region with region which we want to swap
# Creates a dense multi-dimensional meshgrid
H, W = np.mgrid[:spare_image.shape[0], :spare_image.shape[1]]

# Filters out coordinates which color value equals to zeros, that means black region
w = W[spare_image[..., 2] == 0]
h = H[spare_image[..., 2] == 0]

# Fills black region with color from perspective transform result
for i in range(len(w)):
    spare_image[h[i], w[i]] = persp_transform[h[i], w[i]]

del persp_transform
# del dst_image

################ PART TWO - MOVE SECOND REGION TO ANOTHER REGION ################

# Applies a perspective transformation to an image
persp_transform = cv2.warpPerspective(image, np.linalg.inv(homo_matrix),
                        (image.shape[1], image.shape[0]))

# Draws a filled convex polygon with black color
dst_image = cv2.fillConvexPoly(spare_image, src_points.astype(int), 0, 16)

# Fills the black region with region which we want to swap
# Creates a dense multi-dimensional meshgrid
H, W = np.mgrid[:spare_image.shape[0], :spare_image.shape[1]]

# Filters out coordinates which color value equals to zeros, that means black region
w = W[dst_image[..., 2] == 0]
h = H[dst_image[..., 2] == 0]

# Fills black region with color from perspective transform result
for i in range(len(w)):
    dst_image[h[i], w[i]] = persp_transform[h[i], w[i]]

plt.figure(figsize=(10, 8))
plt.title("Output")
plt.imshow(cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB))
plt.show()