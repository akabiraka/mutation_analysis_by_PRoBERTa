import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import torch
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens

class MutDataset(torch.utils.data.Dataset):
    def __init__(self, df, class_dict, max_len=512) -> None:
        super(MutDataset, self).__init__()
        self.df = df 
        self.max_len = max_len
        self.class_dict = class_dict
        self.roberta = RobertaModel.from_pretrained(model_name_or_path="data/pretrained_models/", 
                                                          checkpoint_file="checkpoint_best.pt",
                                                          bpe="sentencepiece", 
                                                          sentencepiece_model="data/bpe_model/m_reviewed.model")
        

    def __len__(self):
        return len(self.df)

    def __transform(self, tokens):
        tokens = tokens[:self.max_len] # truncate
        tokens = collate_tokens([tokens], pad_idx=1, pad_to_length=self.max_len).squeeze(0) # pad 1 at the end
        return tokens

    def __getitem__(self, idx):
        wild_col, mut_col, label_col=0,1,2
        wild_tokens, mut_tokens, label = self.df.loc[idx, wild_col], self.df.loc[idx, mut_col], self.df.loc[idx, label_col]
        wild_seq_ids = self.__transform(self.roberta.encode(wild_tokens))
        mut_seq_ids = self.__transform(self.roberta.encode(mut_tokens))

        wild_seq_ids = torch.tensor(wild_seq_ids, dtype=torch.long)
        mut_seq_ids = torch.tensor(mut_seq_ids, dtype=torch.long)
        
        # making ground-truth class tensor
        class_id = self.class_dict[self.df.loc[idx, label_col]]
        label = torch.tensor(class_id, dtype=torch.long)

        # print(wild_seq_ids.shape, mut_seq_ids.shape, label)
        return wild_seq_ids, mut_seq_ids, label


# example usage
# import pandas as pd

# label_col, fold_col = 2, 3
# fold_no = 1
# df = pd.read_csv("data/splits/all.cv", header=None)
# class_dict = {label:i for i, label in enumerate(df[label_col].unique())}
# # print(class_dict)


# train_df, val_df = df[(df[fold_col]!=fold_no)], df[(df[fold_col]!=fold_no)]
# train_df.reset_index(drop=True, inplace=True)
# # val_df.reset_index(drop=True, inplace=True)
# # print(train_df)

# d = MutDataset(train_df, class_dict)
# d.__getitem__(5)