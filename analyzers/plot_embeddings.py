import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import time
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_pca(features, out_file_path):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features)
    print("PCA:", pca.explained_variance_ratio_)

    pca_one = pca_result[:, 0]
    pca_two = pca_result[:, 1] 
    pca_three = pca_result[:, 2]

    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(xs=pca_one, ys=pca_two, zs=pca_three, c=pca_three, cmap='Greens')
    ax.set_xlabel('Component-one')
    ax.set_ylabel('Component-two')
    ax.set_zlabel('Component-three')
    plt.savefig(out_file_path, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
    plt.close()


def plot_tsne(features, out_file_path):
    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    tsne_one = tsne_results[:,0]
    tsne_two = tsne_results[:,1]
    tsne_three = tsne_results[:,2]

    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(xs=tsne_one, ys=tsne_two, zs=tsne_three, c=tsne_three, cmap='Greens')
    ax.set_xlabel('Component-one')
    ax.set_ylabel('Component-two')
    ax.set_zlabel('Component-three')
    plt.savefig(out_file_path, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
    plt.close()


