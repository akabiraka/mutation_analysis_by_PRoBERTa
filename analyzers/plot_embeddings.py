import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import time
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_pca(features, out_file_path, annotations=None, save=False):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features)
    print("PCA:", pca.explained_variance_ratio_)

    pca_one = pca_result[:, 0]
    pca_two = pca_result[:, 1] 
    pca_three = pca_result[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=pca_one, ys=pca_two, zs=pca_three, c=pca_three, marker=".")
    ax.set_xlabel('Component-one')
    ax.set_ylabel('Component-two')
    ax.set_zlabel('Component-three')
    
    if save:
        plt.savefig(out_file_path, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
        plt.close()
    else:
        plt.show()


def plot_tsne(features, out_file_path, save=False):
    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    tsne_one = tsne_results[:,0]
    tsne_two = tsne_results[:,1]
    tsne_three = tsne_results[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=tsne_one, ys=tsne_two, zs=tsne_three, c=tsne_three)
    ax.set_xlabel('Component-one')
    ax.set_ylabel('Component-two')
    ax.set_zlabel('Component-three')
    if save:
        plt.savefig(out_file_path, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
        plt.close()
    else:
        plt.show()



def plot_multi_pca(features, out_dir, out_prefix, annotations, n_items=10000, incr_amt=500, save=False):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features)
    print("PCA:", pca.explained_variance_ratio_)

    pca_one = pca_result[:, 0]
    pca_two = pca_result[:, 1] 
    pca_three = pca_result[:, 2]

    j=0
    while j!=n_items:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        print(j)
        for i in range(j, j+incr_amt): # plot each point + it's index as text above
            x = pca_one[i]
            y = pca_two[i]
            z = pca_three[i]
            label = annotations[i]
            ax.scatter(x, y, z, c=z, marker=".", alpha=0.0)
            ax.text(x, y, z, label, size=5)
        j=j+incr_amt    
        if save:
            plt.savefig(out_dir+out_prefix+"_"+str(j)+".pdf", dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
            plt.close()
        else:
            plt.show()
            # plt.close()
        # break

def plot_multi_tsne(features, out_dir, out_prefix, annotations, n_items=10000, incr_amt=500, save=False):
    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    tsne_one = tsne_results[:,0]
    tsne_two = tsne_results[:,1]
    tsne_three = tsne_results[:,2]

    j=0
    while j!=n_items:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        print(j)
        for i in range(j, j+incr_amt): # plot each point + it's index as text above
            x = tsne_one[i]
            y = tsne_two[i]
            z = tsne_three[i]
            label = annotations[i]
            ax.scatter(x, y, z, c=z, marker=".", alpha=0.0)
            ax.text(x, y, z, label, size=5)
        j=j+incr_amt    
        if save:
            plt.savefig(out_dir+out_prefix+"_"+str(j)+".pdf", dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
            plt.close()
        else:
            plt.show()
            # plt.close()
        # break
