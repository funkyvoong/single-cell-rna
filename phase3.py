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

'''
@author Pujan Joshi
@date Feb 01, 2019
This code run PCA and tSNE.
This code also draws the plot.
This code also write to the files.
User configurations below.
'''
##User Parameters
RUN_PCA = True
RUN_TSNE = True
SHOW_PLOT = True
WRITE_TO_FILE = True
n_neighbors = 30
n_iter = 700
#input_file_name = 'GSE115469_P5.csv'
#input_file_name = 'GSM2656501_dropseq_1.csv'
#input_file_name = 'xab.csv'
input_file_name = 'GSE124061_MeA_AllCells_DGE.csv'
##Do not change anything below this
t = round(time())

## Loading and curating the data
digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape


## Function to Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], ".",
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# GSE115469_P5.csv
# P5TLH.v2.trimmed4.csv
gaba_names = []
df = pandas.read_csv(input_file_name)
print("done load")

gene_ids = df.iloc[:,0]
df.index = gene_ids

"""
gene_ids = list(gene_ids)
with open("geneset.txt", "r") as f:
    for gene in f:
        if gene.rstrip('\n') in gene_ids:
            gaba_names.append(gene.rstrip('\n'))
"""

cols = list(df.columns)[1:3266] # 3266
#cols = list(df.columns)
#gaba_names = ["5033430I15Rik","5330413P13Rik","5330416C01Rik","5330417C22Rik","5330426P16Rik"]
#gaba_names = ["Brs3",  "Greb1", "Fam84a", "Ankrd55", "Cck", "Nos1"]
#["Gal", "Esr1", "Vstm5", "Robo1", "Cbln2", "Cacna1c","Kcnip4", "Gpr75", "Gpr176","Galr1" ]
#print(gaba_names)
#sml_set = df.loc[gaba_names]
#X = sml_set.values
X = df.values
X = X[:, 1:3266] # 3266
#X = np.transpose(X)

#adata = sc.read_csv(input_file_name)
adata = sc.AnnData(X)
# normalize and filter
sc.pp.normalize_per_cell(adata, copy=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=50)

# pca
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='n_genes')

# tsne
sc.tl.tsne(adata)
sc.pl.tsne(adata, color='n_genes')

# umap
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.pl.umap(adata, color='n_genes')

# louvain
#sc.tl.louvain(adata)
#sc.pl.umap(adata, color=['louvain'], use_raw=False)



# Using sklearn kmeans
km = KMeans(n_clusters=5)
km = km.fit(X)
y_kmeans = km.predict(X)
plt.figure
plt.scatter(X[:,0], X[:,1], c=km.labels_, cmap='rainbow')
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

X = np.transpose(X)

if RUN_PCA:
    ## Computing PCA
    print("Computing PCA projection")
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    tt_pca = round(time() - t0, 2)
    print("time taken by PCA:", tt_pca, "seconds")
    if SHOW_PLOT:
        print("generating PCA plot...")
        plot_title = "Principal Components projection (time %.2fs)" % (tt_pca)
        plot_embedding(X_pca, plot_title)

if RUN_TSNE:
    ## Computing t-SNE
    print("Computing t-SNE with n_iter", n_iter)
    tsne = manifold.TSNE(init='pca', random_state=0, n_iter=n_iter)
    # tsne = manifold.TSNE()
    t0 = time()
    print("Transformation..")
    X_tsne = tsne.fit_transform(X)
    tt_tsne = round(time() - t0, 2)
    print("time taken by tSNE:", tt_tsne, "seconds")
    if SHOW_PLOT:
        print("generating t-SNE plot...")
        plot_title = "t-SNE (n_iter %d)(time %.2fs)" % (n_iter, tt_tsne)
        plot_embedding(X_tsne, plot_title)

if WRITE_TO_FILE:
    print("writing results to file")
    if RUN_PCA or RUN_TSNE:
        cols_t = np.array([cols])
        if RUN_TSNE:
            tsne_filename = 'tsne_' + str(t) + '.txt'
            print("writing tSNE results to file named", tsne_filename)
            X_tsne_2 = np.concatenate((cols_t.T, X_tsne), axis=1)
            np.savetxt(tsne_filename, X_tsne_2, delimiter="\t", fmt='%s')
        if RUN_PCA:
            pca_filename = 'pca_' + str(t) + '.txt'
            print("writing PCA results to file named", pca_filename)
            X_pca_2 = np.concatenate((cols_t.T, X_pca), axis=1)
            np.savetxt(pca_filename, X_pca_2, delimiter="\t", fmt='%s')

if SHOW_PLOT:
    print("displaing plots now ..")
    plt.show()


