import cv2, os
import numpy as np
import matplotlib.pyplot as plt


# Gets current working directory
current_path = os.getcwd()

# Creates output folder
output_path = "output"
os.makedirs(output_path, exist_ok=True)

# Set input image name
image_name = "ArtGallery.jpg"

# Sets input image path
image_path = os.path.join(current_path, image_name)

# Read input image with cv2
image = cv2.imread(image_path)

# Shows image
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# Creates a copy of original input image
spare_image = image.copy()

# Defines source points of first region
src_points = [[38, 102], [341, 124], [342, 426], [42, 447]]

# Defines destination points of second region
dst_points = [[683, 119], [956, 68], [955, 515], [678, 452]]

# Calculates the homography matrix based on SVD
A = []
for i in range(len(src_points)):
    # Get each x, y of source points
    x, y = src_points[i][0], src_points[i][1]
    # Get each x' (u), y' (v) of destinatin points
    u, v = dst_points[i][0], dst_points[i][1]
    A.append([0,   0,   0,   -x,   -y,   -1,   v*x,   v*y,   v])
    A.append([x,   y,   1,    0,    0,    0,  -u*x,  -u*y,  -u])

# Transforms to array to do matrix decomposition
A = np.asarray(A)

# Does the SVD decomposition
U, S, V = np.linalg.svd(A)

# Normalize the output and reshape to matrix with shape 3x3
homo_matrix = (V[-1, :] / V[-1, -1]).reshape(3, 3)

################ PART ONE - MOVE FIRST REGION TO ANOTHER REGION ################

# Applies a perspective transformation to an image
persp_transform = cv2.warpPerspective(spare_image, homo_matrix, 
                        (spare_image.shape[1], spare_image.shape[0]))

# Draws a filled convex polygon with black color
spare_image = cv2.fillConvexPoly(spare_image, np.asarray(dst_points).astype(int), 0, 16)

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
dst_image = cv2.fillConvexPoly(spare_image, np.asarray(src_points).astype(int), 0, 16)

# Fills the black region with region which we want to swap
# Creates a dense multi-dimensional meshgrid
H, W = np.mgrid[:spare_image.shape[0], :spare_image.shape[1]]

# Filters out coordinates which color value equals to zeros, that means black region
w = W[dst_image[..., 2] == 0]
h = H[dst_image[..., 2] == 0]

# Fills black region with color from perspective transform result
for i in range(len(w)):
    dst_image[h[i], w[i]] = persp_transform[h[i], w[i]]

cv2.imwrite(f"{output_path}\\M10907803.jpg", dst_image)

demo = np.concatenate([image, dst_image], axis=1)
cv2.imwrite(f"{output_path}\\demo.jpg", demo)

plt.figure(figsize=(10, 8))
plt.title("Output")
plt.imshow(cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB))
plt.show()