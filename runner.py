from clustering_gradient_interpolation import bilinear_interpolation, bilinear_cluster_interpolation
import os
import numpy as np
import dippykit as dip
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import dippykit as dip
import matplotlib.pyplot as plt
import numpy as np
import MyGMMs
from PIL import Image

root_dir = os.getcwd()
downsampled_path = "11_grayscale_resize\\Level_4"
save_path = "upsample"
hierarchal_path = "hierarchal_clustering"
kmean_path = "kmeans"
gmm_path = "gmm"
downsampled_dir = os.path.join(root_dir, downsampled_path)
save_dir = os.path.join(root_dir, save_path)
background_directories = os.listdir(downsampled_path)
upsample_scale = 2

for background in background_directories:
    image_dir = os.listdir(downsampled_dir + "\\" + background + "\\DSLR_JPG")
    for image_path in image_dir:
        image1 = dip.image_io.im_read(downsampled_dir + "\\" + background + "\\DSLR_JPG\\" + image_path)
        M, N = image1.shape
        new_M = upsample_scale * M
        new_N = upsample_scale * N

        image1_2d = image1.reshape((-1, 1))

        kmeans_cluster = KMeans(n_clusters=8)
        kmeans_cluster.fit(image1_2d)
        cluster_centers = kmeans_cluster.cluster_centers_
        cluster_labels = kmeans_cluster.labels_

        gmm_cluster = GaussianMixture(n_components=2, n_init=20, max_iter=100)
        gmm_cluster.fit(image1_2d)
        cluster_centers_gmm = gmm_cluster.predict(image1_2d)

        # Display the clustered image
        cluster_image_gmm = cluster_centers_gmm.reshape(image1.shape)
        cluster_image_kmeans = cluster_centers[cluster_labels].reshape(image1.shape)
        image_bilinear = bilinear_interpolation(image1, new_M, new_N)
        improve_image_kmeans = bilinear_cluster_interpolation(image1, cluster_image_kmeans, new_M, new_N)
        improve_image_gmm = bilinear_cluster_interpolation(image1, cluster_image_gmm, new_M, new_N)
        improve_image_kmeans = bilinear_cluster_interpolation(image1, cluster_image_kmeans, new_M, new_N)
        # improve_image_hierarch = bilinear_cluster_interpolation(image1, cluster_image_hierarch, new_M, new_N)

        im = Image.fromarray((image_bilinear * 255).astype(np.uint8))
        im.save(save_dir + "\\" + background + "\\bilinear\\bilinear_"+image_path)
        im = Image.fromarray((improve_image_kmeans * 255).astype(np.uint8))
        im.save(save_dir + "\\" + background + "\\kmeans\\kmeans_" + image_path)
        im = Image.fromarray((improve_image_gmm * 255).astype(np.uint8))
        im.save(save_dir + "\\" + background + "\\gmm\\gmm_" + image_path)
        # im = Image.fromarray((improve_image_hierarch * 255).astype(np.uint8))
        # im.save(save_dir + "/hierarchal_clustering" + image_path)