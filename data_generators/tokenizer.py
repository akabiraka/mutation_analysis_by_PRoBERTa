import re
import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import os
from Bio import SeqIO
import sentencepiece as spm


uniprotkb_swissprot_path = "data/uniprotkb_swissprot/uniprotkb_swissprot.fasta"
tokenizer_pretraining_data_path = "data/tokenizer_pretraining_data.txt"
tokenizer_model_prefix = "outputs/models/spm_tokenizer"
tokenizer_model_path = tokenizer_model_prefix+".model"

def filter_seqs(force=False):
    if os.path.exists(uniprotkb_swissprot_path) and force==False: 
        print("sequences are already filtered on length ...")    
        return
    
    print("filtering sequences on length ...")
    seq_list = [record.seq for record in SeqIO.parse(uniprotkb_swissprot_path, "fasta") if len(record.seq)<1024]
    with open(tokenizer_pretraining_data_path, 'w') as filehandle:
        for seq in seq_list:
            filehandle.write('%s\n' % seq) 
            
def train_tokenizer():
    if os.path.exists(tokenizer_model_path):
        print("sequences are already tokenized")
        return
    spm.SentencePieceTrainer.Train("--input={} --model_prefix={} --vocab_size=10000 --character_coverage=1.0\
        --model_type=bpe --max_sentence_length=1024".format(tokenizer_pretraining_data_path, tokenizer_model_prefix))
    
if __name__ == "__main__":
    filter_seqs()
    train_tokenizer()
    model = spm.SentencePieceProcessor()
    model.load(tokenizer_model_path)