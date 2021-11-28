import numpy as np
import pandas as pd
import os
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt     
from fairseq.models.roberta import RobertaModel

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


pretrained_model_dir="data/pretrained_models/"
vocab_path="data/pretrained_models/dict.txt"
sentencepiece_model_path="data/bpe_model/m_reviewed.model"

vocab_col=0
freq_col=1
vocab_df = pd.read_csv(vocab_path, header=None, delimiter=" ")
print(vocab_df.shape)
#print(vocab_df.head())

all_vocab_features = get_all_vocab_features(force=False)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(all_vocab_features)
print(pca.explained_variance_ratio_)

pca_one = pca_result[:, 0]
pca_two = pca_result[:, 1] 
pca_three = pca_result[:, 2]


ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(xs=pca_one, ys=pca_two, zs=pca_three, c=pca_three, cmap='Greens')
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.savefig("outputs/images/vocab_embedding.pdf", dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)

