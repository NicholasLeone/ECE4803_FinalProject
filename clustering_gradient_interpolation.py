from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import dippykit as dip
import matplotlib.pyplot as plt
import numpy as np
import MyGMMs
import torch
from torch import nn
from PIL import Image

def bilinear_cluster_interpolation(image, cluster, new_height, new_width):
    # Get the original image dimensions
    pad = 4
    height, width = image.shape
    image_pad = np.pad(image, (pad, pad), mode='edge')
    # Ix_pad = np.pad(Ix, (pad, pad), mode='edge')
    # Iy_pad = np.pad(Iy, (pad, pad), mode='edge')
    cluster_pad = np.pad(cluster, (pad, pad), mode='edge')

    # Create an output image with the desired dimensions
    interpolated_image = np.zeros((new_height, new_width), dtype=float)

    # Calculate scaling factors
    x_scale = float(width) / new_width
    y_scale = float(height) / new_height
    four_count = 0

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
            source_pixels = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
            source_labels = np.array([cluster[y0, x0], cluster[y0, x1], cluster[y1, x0], cluster[y1, x1]])
            source_image = np.array([image[y0, x0], image[y0, x1], image[y1, x0], image[y1, x1]])

            # source_Ix = np.array([Ix_pad[x0+pad, y0+pad], Ix_pad[x1+pad, y0+pad], Ix_pad[x0+pad, y1+pad], Ix_pad[x1+pad, y1+pad]])
            # source_Iy = np.array([Iy_pad[x0 + pad, y0 + pad], Iy_pad[x1 + pad, y0 + pad], Iy_pad[x0 + pad, y1 + pad], Iy_pad[x1 + pad, y1 + pad]])
            # source_dist = np.array([x_original, y_original]) - source_pixels
            # source_gradient = source_image + np.multiply(source_dist[:, 0], source_Ix) + np.multiply(source_dist[:, 1], source_Iy)
            # source_gradient[source_gradient > 1] = 1
            # source_gradient[source_gradient < 0] = 0

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
            if np.unique(source_labels).shape[0] > 1:
                arg_ind = dist_weights.argsort()[::-1]
                # print(dist_weights[arg_ind])
                sort_labels = source_labels[arg_ind]
                # sort_pixels = source_pixels[arg_ind]
                sort_image = source_image[arg_ind]
                # sort_gradient = source_gradient[arg_ind]
                sort_weights = dist_weights[arg_ind]
                # new_source_pixels = np.zeros((4,))
                main_label = np.argwhere(sort_labels[0] == sort_labels)
                sub_label = np.argwhere(sort_labels[0] != sort_labels)
                sort_weights[main_label] = np.sqrt(sort_weights[main_label])
                sort_weights[sub_label] = sort_weights[sub_label] ** 2
                # interp_labels = sort_labels[arg_label]
                # interp_image = sort_image[arg_label]
                # interp_weights = sort_weights[arg_label]
                # interp_gradient = sort_gradient[arg_label]

                if sort_weights.sum() > 0 and sort_weights.shape[0] != 1:
                    # print("Sort_Label: ", sort_labels)
                    # print("Sort_weights: ", sort_weights)
                    # print("Interp Weights: ", interp_weights)
                    sort_weights = sort_weights / sort_weights.sum()
                interpolated_image[y, x] = np.sum(np.multiply(sort_image, sort_weights))
            else:
                interpolated_image[y, x] = np.sum(np.multiply(source_image, dist_weights))


            # print("Inter-Cluster Pre: ", source_image)


    return interpolated_image

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
            source_pixels = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])

            # Calculate the interpolation weights
            wx1 = x_original - x0
            wy1 = y_original - y0
            wx0 = 1 - wx1
            wy0 = 1 - wy1

            interpolated_value = (
                    wx0 * wy0 * image[y0, x0] +
                    wx1 * wy0 * image[y0, x1] +
                    wx0 * wy1 * image[y1, x0] +
                    wx1 * wy1 * image[y1, x1])
            interpolated_image[y, x] = interpolated_value

    return interpolated_image

# image = dip.image_io.im_read("2_5_1_016_10_0.jpg")
# image_pad = np.pad(image, (3,3), mode='symmetric')
# M, N = image.shape
# print(image.shape)
# image_2d = image.reshape((-1, 1))
#
# M1 = np.array([[4, 0], [0, 4]])
# image1 = dip.resample(image, M1)
# image1_pad = dip.resample(image_pad, M1)
# image1_2d = image1.reshape((-1, 1))
# # intensity_values = image1.flatten().reshape((-1,1))
# # # Perform hierarchical clustering
# # linked = linkage(intensity_values, method='ward', metric='euclidean')
# #
# # threshold = 0.5  # Adjust as needed
# # cluster_labels = fcluster(linked, threshold, criterion='distance')
# # # Reshape the cluster labels to match the original image shape
# # clustered_image = cluster_labels.reshape(image1.shape)
# # print(np.unique(cluster_labels))
#
# kmeans_cluster = KMeans(n_clusters=8)
# kmeans_cluster.fit(image1_2d)
# cluster_centers = kmeans_cluster.cluster_centers_
# cluster_labels = kmeans_cluster.labels_
#
# # gmm_cluster = GaussianMixture(n_components=2, n_init=20, max_iter=100)
# # gmm_cluster.fit(image1_2d)
# # cluster_centers_gmm = gmm_cluster.predict(image1_2d)
#
# # Display the clustered image
# # cluster_image_gmm = cluster_centers_gmm.reshape(image1.shape)*255
# cluster_image_kmeans = cluster_centers[cluster_labels].reshape(image1.shape)
# # print(np.unique(cluster_image_kmeans))
# # Ix, Iy = compute_image_gradients(image1)
# image_bilinear = bilinear_interpolation(image1, M, N)
# improve_image = bilinear_cluster_interpolation(image1, cluster_image_kmeans, M, N)
# M1 = np.array([[4, 0], [0, 4]])
# image_step = dip.resample(image1_pad, np.linalg.inv(M1), crop=True, crop_size=(M, N), interpolation='nearest')
#
# print(dip.metrics.PSNR(image, image_bilinear, max_signal_value=1.0))
# print(dip.metrics.PSNR(image, improve_image, max_signal_value=1.0))
#
# SSIM_bilinear, SSIM_bilinear_image = dip.metrics.SSIM(image, image_bilinear, data_range=1.0)
# SSIM_improve, SSIM_improve_image = dip.metrics.SSIM(image, improve_image, data_range=1.0)
#
# print("SSIM_Bilinear: ", SSIM_bilinear)
# print("SSIM_Improve: ", SSIM_improve)
#
# print("Improve Mean ", np.mean(improve_image))
# print("Bilinear Mean", np.mean(image_bilinear))
#
# im = Image.fromarray((image_bilinear * 255).astype(np.uint8))
# im.save("image_bilinear.jpg")
#
# im = Image.fromarray((improve_image * 255).astype(np.uint8))
# im.save("improve_image.jpg")
# # fig = plt.figure(figsize = (15,8))
# plt.figure()
# plt.imshow(improve_image)
# #plt.imshow(cluster_centers[cluster_labels].reshape(image.shape))
# plt.title('Gradient-Interp Image')
# plt.axis('off')
#
# plt.figure()
# plt.imshow(image_bilinear)
# plt.title('Bilinear Interp')
# plt.axis('off')
#
# plt.figure()
# plt.imshow(SSIM_bilinear_image)
# plt.title('SSIM Bilinear Interp')
# plt.axis('off')
#
# plt.figure()
# plt.imshow(SSIM_improve_image)
# plt.title('SSIM Improve Interp')
# plt.axis('off')
#
# plt.figure()
# plt.imshow(image)
# plt.title('Original')
# plt.axis('off')
# plt.show()