import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import os
import pandas as pd
from Bio import SeqIO
import sentencepiece as spm


uniprotkb_swissprot_path = "data/uniprotkb_swissprot/uniprotkb_swissprot.fasta"
pretraining_file = "data/pretraining_data_for_tokenizer.txt"
pretraining_tokenized_file = "data/pretraining_data_tokenized.txt"

tokenizer_model_prefix = "outputs/models/spm_tokenizer"
tokenizer_model_path = tokenizer_model_prefix+".model"

fastas_dir = "data/fastas/"
train_dataset_path = "data/dataset_5_train.csv"
tokenized_dir = "data/tokenized/"



def filter_seqs(force=False):
    if os.path.exists(uniprotkb_swissprot_path) and force==False: 
        print("sequences are already filtered on length ...")    
        return
    
    print("filtering sequences on length ...")
    seq_list = [record.seq for record in SeqIO.parse(uniprotkb_swissprot_path, "fasta") if len(record.seq)<1024]
    with open(pretraining_file, 'w') as filehandle:
        for seq in seq_list:
            filehandle.write('%s\n' % seq) 
            
def train_tokenizer():
    if os.path.exists(tokenizer_model_path):
        print("sequences are already tokenized")
        return
    spm.SentencePieceTrainer.Train("--input={} --model_prefix={} --vocab_size=10000 --character_coverage=1.0\
        --model_type=bpe --max_sentence_length=1024".format(pretraining_file, tokenizer_model_prefix))
    
def tokenize_pretraining_seqs(model):
    for record in SeqIO.parse(uniprotkb_swissprot_path, "fasta"):
        if len(record.seq)<1024:
            toked = model.encode_as_pieces(str(record.seq))
            with open(pretraining_tokenized_file, 'a') as filehandle:
                    filehandle.write('%s\n' % " ".join(toked))
            
def tokenize_mutation_seqs(model, dataset_path):
    df = pd.read_csv(dataset_path)
    train_tokenized_file_path = tokenized_dir+"train.sequence"
    if os.path.exists(train_tokenized_file_path):
        os.remove(train_tokenized_file_path)
    for i, row in df.iterrows():
        wild_fasta_path = fastas_dir+row["pdb_id"]+row["chain_id"]+".fasta"
        wild_seq = [record.seq for record in SeqIO.parse(wild_fasta_path, "fasta")][0]
        wild_tokened = model.encode_as_pieces(str(wild_seq))
        wild_tokened = " ".join(wild_tokened)
        
        mutant_fasta_path = fastas_dir+row["pdb_id"]+row["chain_id"]+"_"+row["mutation"]+".fasta"
        mutant_seq = [record.seq for record in SeqIO.parse(mutant_fasta_path, "fasta")][0]
        mutant_tokened = model.encode_as_pieces(str(mutant_seq))
        mutant_tokened = " ".join(mutant_tokened)
        
        with open(train_tokenized_file_path, "a") as f:
            f.write(wild_tokened + " <sep> " + mutant_tokened +"\n")
            f.write(mutant_tokened + " <sep> " + wild_tokened +"\n")
            
        # if i==5:
        #     break
    
if __name__ == "__main__":
    filter_seqs()
    train_tokenizer()
    model = spm.SentencePieceProcessor()
    model.load(tokenizer_model_path)
    # tokenize_pretraining_seqs(model)
    tokenize_mutation_seqs(model, dataset_path=train_dataset_path)