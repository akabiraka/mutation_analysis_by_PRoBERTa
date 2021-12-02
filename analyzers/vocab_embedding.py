import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import numpy as np
import pandas as pd
import os
import time
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt     
from fairseq.models.roberta import RobertaModel
from analyzers.plot_embeddings import *



def get_all_vocab_features(force=False):
    out_file="outputs/npy_data/all_vocab_features.npy"
    if os.path.exists(out_file) and force==False:
        with open(out_file, "rb") as f: return np.load(f)
        
    roberta_model = RobertaModel.from_pretrained(model_name_or_path=pretrained_model_dir, checkpoint_file="checkpoint_best.pt", bpe="sentencepiece", sentencepiece_model=sentencepiece_model_path)
    roberta_model.eval()
    #print(roberta_model)
    
    all_vocab_features = []
    for i, row in enumerate(vocab_df.itertuples(index=False)):
        vocab=str(row[vocab_col]).strip()
        print(vocab)
        encoded = roberta_model.encode(vocab)
        decoded = roberta_model.decode(encoded)
        print(encoded, decoded)
        
        last_layer_features = roberta_model.extract_features(encoded)
        print("last_layer_feature_size: ", last_layer_features.size()) #(1, 4, 768)
        
        vocab_features = last_layer_features.sum(dim=1).squeeze().detach().numpy() #786
        all_vocab_features.append(vocab_features)
        
        print()
        #if i==5: break
    
    with open(out_file, "wb") as f: np.save(f, np.array(all_vocab_features))
    with open(out_file, "rb") as f: return np.load(f)


# def plot_multi_pca(features, out_dir, out_prefix, annotations=None, save=False):
#     pca = PCA(n_components=3)
#     pca_result = pca.fit_transform(features)
#     print("PCA:", pca.explained_variance_ratio_)

#     pca_one = pca_result[:, 0]
#     pca_two = pca_result[:, 1] 
#     pca_three = pca_result[:, 2]

#     j=0
#     while j!=10000:
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')
#         print(j)
#         n_vocabs=500
#         for i in range(j, j+n_vocabs): # plot each point + it's index as text above
#             x = pca_one[i]
#             y = pca_two[i]
#             z = pca_three[i]
#             label = annotations[i]
#             ax.scatter(x, y, z, c=z, marker=".", alpha=0.0)
#             ax.text(x, y, z, label, size=5)
#         j=j+n_vocabs    
#         if save:
#             plt.savefig(out_dir+out_prefix+"_"+str(j)+".pdf", dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
#             plt.close()
#         else:
#             plt.show()
#             # plt.close()
#         # break

# def plot_multi_tsne(features, out_dir, out_prefix, annotations=None, save=False):
#     time_start = time.time()
#     tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
#     tsne_results = tsne.fit_transform(features)
#     print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

#     tsne_one = tsne_results[:,0]
#     tsne_two = tsne_results[:,1]
#     tsne_three = tsne_results[:,2]

#     j=0
#     while j!=10000:
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')
#         print(j)
#         n_vocabs=500
#         for i in range(j, j+n_vocabs): # plot each point + it's index as text above
#             x = tsne_one[i]
#             y = tsne_two[i]
#             z = tsne_three[i]
#             label = annotations[i]
#             ax.scatter(x, y, z, c=z, marker=".", alpha=0.0)
#             ax.text(x, y, z, label, size=5)
#         j=j+n_vocabs    
#         if save:
#             plt.savefig(out_dir+out_prefix+"_"+str(j)+".pdf", dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
#             plt.close()
#         else:
#             plt.show()
#             # plt.close()
#         # break

pretrained_model_dir="data/pretrained_models/"
vocab_path="data/pretrained_models/dict.txt"
sentencepiece_model_path="data/bpe_model/m_reviewed.model"

vocab_col=0
freq_col=1
vocab_df = pd.read_csv(vocab_path, header=None, delimiter=" ")
print(vocab_df.shape)
print(vocab_df.head())


all_vocab_features = get_all_vocab_features(force=False)

plot_pca(all_vocab_features, "outputs/images/embeddings/vocab_pca_embed.pdf", save=True)
plot_tsne(all_vocab_features, "outputs/images/embeddings/vocab_tsne_embed.pdf", save=True)


annotations=vocab_df[0].tolist()
print(annotations)
plot_multi_pca(all_vocab_features, "outputs/images/vocab_pca/", "embedding", annotations=annotations, n_items=10000, incr_amt=500, save=True)
plot_multi_tsne(all_vocab_features, "outputs/images/vocab_tsne/", "embedding", annotations=annotations, n_items=10000, incr_amt=500, save=True)

