import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import sentencepiece as spm
from fairseq.models.roberta import RobertaModel
from analyzers.plot_embeddings import *

# inp/out configs 
tokenized_dir = "data/bpe_tokenized/"
fastas_dir = "data/fastas/"
train_dataset_path = "data/dataset_5_train.csv"
#validation_dataset_path = "data/dataset_5_validation.csv"
#test_dataset_path = "data/dataset_5_test.csv"

out_file_path="data/bpe_tokenized/train.csv"
 

#def init_models():
# bpe specific
tokenizer_model_path = "data/bpe_model/m_reviewed.model"
spm_model = spm.SentencePieceProcessor()
spm_model.load(tokenizer_model_path)
    
# roberta specific
pretrained_model_dir="data/pretrained_models/"
roberta_model = RobertaModel.from_pretrained(model_name_or_path=pretrained_model_dir, checkpoint_file="checkpoint_best.pt", bpe="sentencepiece", sentencepiece_model=tokenizer_model_path)
roberta_model.eval()


def tokenize_seq(seq):    
    toked = spm_model.encode_as_pieces(str(seq))
    return " ".join(toked)


def get_features(fasta_path):
    seq = [record.seq for record in SeqIO.parse(fasta_path, "fasta")][0]
    toked = tokenize_seq(seq)
    encoded = roberta_model.encode(toked)
    last_layer_features = roberta_model.extract_features(encoded)
    features = last_layer_features.sum(dim=1).squeeze().detach().numpy()
    return seq, toked, encoded, features
   

def features():
    pdb_ids_df = pd.read_csv("outputs/npy_data/all_proteins.csv", header=None)
    with open("outputs/npy_data/all_protein_features.npy", "rb") as f:
        features = np.load(f)
    return pdb_ids_df, features


def compute_features(inp_dataset_path, force=False):
    if os.path.exists("outputs/npy_data/all_protein_features.npy") and force==False:
        return features()
    
    df = pd.read_csv(inp_dataset_path)
    proteins_set = set()
    all_protein_features = []    
    all_proteins = []
    for i, row in df.iterrows():
        print(i, row)
        label = "stabilizing" if row["ddG"]>=0 else "destabilizing"
        
        wild_pdb_id = row["pdb_id"]+row["chain_id"] 
        wild_seq, wild_toked, wild_encoded, wild_features = get_features(fastas_dir+wild_pdb_id+".fasta")
        #print(wild_seq, wild_toked, wild_encoded, wild_features.shape)

        mutant_pdb_id = row["pdb_id"]+row["chain_id"]+"_"+row["mutation"]
        mutant_seq, mutant_toked, mutant_encoded, mutant_features = get_features(fastas_dir+mutant_pdb_id+".fasta")
        #print(mutant_seq, mutant_toked, mutant_encoded, mutant_features.shape)

        if wild_pdb_id not in proteins_set:
            all_protein_features.append(wild_features)
            all_proteins.append(wild_pdb_id)
            proteins_set.add(wild_pdb_id)

        all_protein_features.append(mutant_features)
        all_proteins.append(mutant_pdb_id)

        print()
        #if i==5: break

    out_df = pd.DataFrame(all_proteins)
    out_df.to_csv("outputs/npy_data/all_proteins.csv", index=False, header=False)
    with open("outputs/npy_data/all_protein_features.npy", "wb") as f: np.save(f, np.array(all_protein_features))
    return features()



if __name__ == "__main__":
    pdb_ids_df, features =compute_features(train_dataset_path, force=False) 
    print(pdb_ids_df.shape, features.shape)
    
    plot_pca(features, "outputs/images/embeddings/seq_pca_embed.pdf", save=True)
    plot_tsne(features, "outputs/images/embeddings/seq_tsne_embed.pdf", save=True)
    
    annotations = pdb_ids_df[0].tolist()
    # print(len(annotations))
    plot_multi_pca(features, "outputs/images/seq_pca/", "embedding", annotations=annotations, n_items=4000, incr_amt=200, save=True)
    plot_multi_tsne(features, "outputs/images/seq_tsne/", "embedding", annotations=annotations, n_items=4000, incr_amt=200, save=True)
    
