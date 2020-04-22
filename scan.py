## importing the required packages
import pandas
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import scanpy as sc
#tsne_1554939182.txt

RUN_KMEANS = True
SHOW_PLOT = True
WRITE_TO_FILE = True
t = round(time())

X = pandas.read_csv('tsne_1554690315.txt', sep="\t", header=None)
cells = X.iloc[:,0]
X.index = cells # set cell names as index

#X.columns = ['cell name','col2','col3']
X.index.name = None
newX = X.T.iloc[1:]
df = np.transpose(newX)


# Using sklearn kmeans
km = KMeans(n_clusters=5)
km = km.fit(df)
y = km.predict(df)

if WRITE_TO_FILE:
    print("writing results to file")
    if RUN_KMEANS:
        #cols_t = np.array([cells])
        kmeans_filename = 'kmeans_' + str(t) + '.txt'
        print("writing kmeaans results to file named", kmeans_filename)
        #print(df)
        #X_kmeans_2 = np.concatenate((df, y.T), axis=1)
        #print(X_kmeans_2)
        np.savetxt(kmeans_filename, y, delimiter="\t", fmt='%s')


"""
plt.figure()
plt.scatter(df[:,0], df[:,1], c=y, cmap='rainbow')
centers = km.cluster_centers_
plt.scatter(centers[:,0], centers[:, 1], c='black')
plt.show()


# using sklearn dbscan
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps=0.123, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='rainbow')
plt.show()
"""