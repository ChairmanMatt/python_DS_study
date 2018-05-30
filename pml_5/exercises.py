import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_inertia(km, X, n_cluster_range):
    inertias = []
    for i in n_cluster_range:
        km.set_params(n_clusters=i)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.plot(n_cluster_range, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


# pic/temp.png is a 24-bit RGB image file
# 24-bit means that 8 bits are allocated to R, G, and B. In other words, each color can have 256 different values
img = mpimg.imread('pic/temp.png')
#plt.figure(figsize=(10, 10))
#imgplot = plt.imshow(img)
#imgplot.axes.axis('off')
#plt.show()

# 3D array
# corresponding to red, blue, green
# all the numbers in dimension 1 and dimension 2 ranges from 0 to 255
img_shape = img.shape
m1, m2, m3 = img_shape
print(img_shape)

data_rs = img.reshape(m1 * m2, m3)
print(data_rs.shape)
print(data_rs.dtype)

# Image data is typically float32 but KMeans will throw an exception if n_clusters > 32
# Solution is to convert image data to float64 for KMeans, and optionally back to float32 to show
data_rs64 = data_rs.astype(np.float64)

from sklearn.cluster import KMeans
import matplotlib.image as mpimg

def KmeansCompression(data, nclus=16, **kmean_kwargs):
    '''
    data: data to cluster
    nclus: number of colors
    '''
    cluster = KMeans(n_clusters=nclus, **kmean_kwargs)
    cluster.fit(data)
    centers = cluster.cluster_centers_
    labels = cluster.labels_
    return centers[labels]

#img128 = KmeansCompression(data_rs64, nclus=128, n_jobs=1)
#img128 = img128.reshape(m1, m2, m3)
#plt.figure(figsize=(10, 10))
#plt.imshow(img128)
#plt.axis('off')
#plt.show()

#img64 = KmeansCompression(data_rs64, nclus=32, n_jobs=1)
#img64 = img64.reshape(*img_shape)
#plt.figure(figsize=(10, 10))
#plt.imshow(img64)
#plt.axis('off')
#plt.show()

# On my machine, data_rs works with 16 colors but you may have to use data_rs64
img_16 = KmeansCompression(data_rs64, nclus=16, n_jobs=1)
img_16 = img_16.reshape(*img_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img_16)
plt.axis('off')
plt.show()

#n, bins, patches = plt.hist(img_16.ravel(), bins=16, range=(0.0, 1.0), fc='k', ec='k')
#plt.show()

#kmeans = KMeans(n_jobs=1)
#plot_inertia(kmeans, data_rs64, [16, 64, 128])