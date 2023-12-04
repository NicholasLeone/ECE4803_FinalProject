from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import dippykit as dip
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numba import cuda, jit

@jit(target_backend='cuda')
def bilinear_cluster_interpolation(image, cluster, new_height, new_width):
    # Get the original image dimensions
    height, width = image.shape

    # Create an output image with the desired dimensions
    interpolated_image = np.zeros((new_height, new_width), dtype=float)

    # Calculate scaling factors
    x_scale = float(width) / new_width
    y_scale = float(height) / new_height

    for y in range(new_height):
        for x in range(new_width):
            # Calculate the corresponding coordinates in the original image
            x_original = x * x_scale
            y_original = y * y_scale

            # Calculate the integer coordinates surrounding the (x_original, y_original) point
            x0 = int(x_original)
            x1 = min(x0 + 1, width - 1)
            y0 = int(y_original)
            y1 = min(y0 + 1, height - 1)

            # Get cluster labels and pixels of the four known points
            source_labels = np.array([cluster[y0, x0], cluster[y0, x1], cluster[y1, x0], cluster[y1, x1]])
            source_image = np.array([image[y0, x0], image[y0, x1], image[y1, x0], image[y1, x1]])

            # Calculate the interpolation weights
            wx1 = x_original - x0
            wy1 = y_original - y0
            wx0 = 1 - wx1
            wy0 = 1 - wy1

            wx0_wy0 = wx0 * wy0
            wx1_wy0 = wx1 * wy0
            wx0_wy1 = wx0 * wy1
            wx1_wy1 = wx1 * wy1
            dist_weights = np.array([wx0_wy0, wx1_wy0, wx0_wy1, wx1_wy1])

            # Adjust weights if all four known points are not in the same cluster.
            if np.unique(source_labels).shape[0] > 1:
                # Get indices of distance weights sorted from greatest to least.
                # The interpolated point will be in the same cluster as the known point with the greatest distance weight.
                arg_ind = dist_weights.argsort()[::-1]

                # Sort labels, pixels, and weights based off of arg_ind
                sort_labels = source_labels[arg_ind]
                sort_image = source_image[arg_ind]
                sort_weights = dist_weights[arg_ind]

                # Get indices of known points that are in the same cluster as the interpolated point.
                main_label = np.argwhere(sort_labels[0] == sort_labels)

                # Get indices of known points that are NOT in the same cluster as the interpolated point.
                sub_label = np.argwhere(sort_labels[0] != sort_labels)

                # Adjust weights of known points that are in the same cluster as the interpolated point.
                sort_weights[main_label] = np.sqrt(sort_weights[main_label])
                # Do not adjust the weights of known points that are not in the same cluster.
                sort_weights[sub_label] = sort_weights[sub_label]

                # Normalize weights such that their sum adds up to 1.
                if sort_weights.sum() > 0 and sort_weights.shape[0] != 1:
                    sort_weights = sort_weights / sort_weights.sum()

                # Calculate the pixel value of the interpolated point with adjusted weights.
                interpolated_image[y, x] = np.sum(np.multiply(sort_image, sort_weights))
            else:
                # Resume normal bilinear interpolation.
                interpolated_image[y, x] = np.sum(np.multiply(source_image, dist_weights))
    return interpolated_image

@jit(target_backend='cuda')
def bilinear_interpolation(image, new_height, new_width):
    # Get the original image dimensions
    height, width = image.shape

    # Create an output image with the desired dimensions
    interpolated_image = np.zeros((new_height, new_width), dtype=float)

    # Calculate scaling factors
    x_scale = float(width) / new_width
    y_scale = float(height) / new_height

    for y in range(new_height):
        for x in range(new_width):
            # Calculate the corresponding coordinates in the original image
            x_original = x * x_scale
            y_original = y * y_scale

            # Calculate the integer coordinates surrounding the (x_original, y_original) point
            x0 = int(x_original)
            x1 = min(x0 + 1, width - 1)
            y0 = int(y_original)
            y1 = min(y0 + 1, height - 1)

            # Get pixels of the four known points
            source_image = np.array([image[y0, x0], image[y0, x1], image[y1, x0], image[y1, x1]])

            # Calculate the interpolation weights
            wx1 = x_original - x0
            wy1 = y_original - y0
            wx0 = 1 - wx1
            wy0 = 1 - wy1

            wx0_wy0 = wx0 * wy0
            wx1_wy0 = wx1 * wy0
            wx0_wy1 = wx0 * wy1
            wx1_wy1 = wx1 * wy1
            dist_weights = np.array([wx0_wy0, wx1_wy0, wx0_wy1, wx1_wy1])

            # Perform bilinear interpolation.
            interpolated_image[y, x] = np.sum(np.multiply(source_image, dist_weights))

    return interpolated_image