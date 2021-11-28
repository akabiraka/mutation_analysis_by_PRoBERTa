from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
import pandas as pd
import numpy as np
import torch

batch_size = 1 
from_col=0
to_col=1
label_col=2
classification_head="mutation_classification"
pretrained_model_dir = "data/pretrained_models/"

data =  pd.read_csv("data/bpe_tokenized/test.from", header=None)
print(data.shape)

split_num=int(len(data) / batch_size)
batched_data=np.array_split(data, split_num)
print("Total batches: " + str(len(batched_data)))

roberta_model = RobertaModel.from_pretrained(model_name_or_path=pretrained_model_dir, checkpoint_file="checkpoint_best.pt", bpe="sentencepiece", sentencepiece_model="data/bpe_model/m_reviewed.model")
roberta_model.eval()
print(roberta_model)

for count, batch_df in enumerate(batched_data):
    for tokens in batch_df.itertuples(index=False):
        print(tokens[from_col])#, tokens[to_col], tokens[label_col])
        encoded = roberta_model.encode(tokens[from_col])#, tokens[to_col])
        print(encoded)
        decoded = roberta_model.decode(encoded)
        print(decoded)
        # torch.ones(512, dtype = torch.long) 
        # batch=collate_tokens([torch.cat((roberta_model.encode(tokens[from_col], tokens[to_col]), torch.ones(512, dtype = torch.long)))[:512]
        #     for tokens in batch_df.itertuples(index=False)], pad_idx=1)
        
        # break
    break
