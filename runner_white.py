from clustering_interpolation import bilinear_interpolation, bilinear_cluster_interpolation
import os
import numpy as np
import dippykit as dip
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import dippykit as dip
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Get working directory
root_dir = os.getcwd()

# Path of downsampled images
downsampled_path = "11_grayscale_resize\\Level_4"

# Path of output images
save_path = "upsample"

# Join root directory and path of downsampled images
downsampled_dir = os.path.join(root_dir, downsampled_path)

# Join root directory and path of output images
save_dir = os.path.join(root_dir, save_path)

# Upsample scaling
upsample_scale = 2

# Background category
background = "white"

# Directory of images in category
image_dir = os.listdir(downsampled_dir + "\\" + background + "\\DSLR_JPG")

# Load one image at a time and run through CABI and Bilinear Interpolation
for image_path in image_dir:
    # Load downsampled image
    image1 = dip.image_io.im_read(downsampled_dir + "\\" + background + "\\DSLR_JPG\\" + image_path)

    # Get shape of downsampled image
    M, N = image1.shape

    # Calculate size of upsampled image
    new_M = upsample_scale * M
    new_N = upsample_scale * N

    # Flatten downsampeld image
    image1_2d = image1.reshape((-1, 1))

    # Initialize and Run K-Means on image
    kmeans_cluster = KMeans(n_clusters=8)
    kmeans_cluster.fit(image1_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_

    # Initialize and run GMM on image
    gmm_cluster = GaussianMixture(n_components=8, n_init=20, max_iter=100)
    gmm_cluster.fit(image1_2d)
    cluster_centers_gmm = gmm_cluster.predict(image1_2d)

    # Run CABI and Bilinear Interpolation
    cluster_image_gmm = cluster_centers_gmm.reshape(image1.shape)
    cluster_image_kmeans = cluster_centers[cluster_labels].reshape(image1.shape)
    image_bilinear = bilinear_interpolation(image1, new_M, new_N)
    improve_image_kmeans = bilinear_cluster_interpolation(image1, cluster_image_kmeans, new_M, new_N)
    improve_image_gmm = bilinear_cluster_interpolation(image1, cluster_image_gmm, new_M, new_N)
    improve_image_kmeans = bilinear_cluster_interpolation(image1, cluster_image_kmeans, new_M, new_N)

    # Save upsampled images to respective folders
    im = Image.fromarray((image_bilinear * 255).astype(np.uint8))
    im.save(save_dir + "\\" + background + "\\bilinear\\bilinear_"+image_path)
    im = Image.fromarray((improve_image_kmeans * 255).astype(np.uint8))
    im.save(save_dir + "\\" + background + "\\kmeans\\kmeans_" + image_path)
    im = Image.fromarray((improve_image_gmm * 255).astype(np.uint8))
    im.save(save_dir + "\\" + background + "\\gmm\\gmm_" + image_path)