import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import os
import pandas as pd
from Bio import SeqIO
import sentencepiece as spm

tokenizer_model_path = "data/bpe_model/m_reviewed.model"
tokenized_dir = "data/bpe_tokenized/"
fastas_dir = "data/fastas/"
train_dataset_path = "data/dataset_5_train.csv"
#validation_dataset_path = "data/dataset_5_validation.csv"
#test_dataset_path = "data/dataset_5_test.csv"

def tokenize_fasta_seq(fasta_path):
    seq = [record.seq for record in SeqIO.parse(fasta_path, "fasta")][0]
    toked = model.encode_as_pieces(str(seq))
    return " ".join(toked)



def tokenize_all_seq(model, dataset_path, prefix, force=False):
    out_file_path = tokenized_dir+prefix+".csv"
    if os.path.exists(out_file_path) and force==False:
        return pd.read_csv(out_file_path)
    
    df = pd.read_csv(dataset_path)
    data = []    
    for i, row in df.iterrows():
        print(i, row)
        label = "stabilizing" if row["ddG"]>=0 else "destabilizing"
        
        wild_pdb_id = row["pdb_id"]+row["chain_id"] 
        wild_toked = tokenize_fasta_seq(fastas_dir+wild_pdb_id+".fasta")
        
        mutant_pdb_id = row["pdb_id"]+row["chain_id"]+"_"+row["mutation"]
        mutant_tokened = tokenize_fasta_seq(fastas_dir+mutant_pdb_id+".fasta")

        data.append([wild_pdb_id, wild_toked, mutant_pdb_id, mutant_tokened, label])
    
    out_df = pd.DataFrame(data)
    out_df.to_csv(out_file_path, index=False)
    return pd.read_csv(out_file_path)



if __name__ == "__main__":
    model = spm.SentencePieceProcessor()
    model.load(tokenizer_model_path)
    df=tokenize_all_seq(model, dataset_path=train_dataset_path, prefix="train") 
    print(df.shape)
    print(df.head())

