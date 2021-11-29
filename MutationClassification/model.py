import sys
from matplotlib.pyplot import cla
sys.path.append("../mutation_analysis_by_PRoBERTa")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

class Net(nn.Module):
    def __init__(self, freeze_pretrained_bert=True, drop_prob=0.5):
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained(model_name_or_path="data/pretrained_models/", checkpoint_file="checkpoint_best.pt", 
                                                     bpe="sentencepiece", sentencepiece_model="data/bpe_model/m_reviewed.model")
        if freeze_pretrained_bert: self.roberta_model.eval()
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 2),
            nn.Dropout(p=drop_prob)
        )
        self.sigmoid = nn.Sigmoid()
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    
    def forward(self, wild_tokens, mut_tokens):
        wild_encoded = self.roberta_model.encode(wild_tokens)
        mut_encoded = self.roberta_model.encode(mut_tokens)
        
        wild_features = self.roberta_model.extract_features(wild_encoded).sum(dim=1).squeeze()
        mut_features = self.roberta_model.extract_features(mut_encoded).sum(dim=1).squeeze()
        # print(wild_features.shape, mut_features.shape)
        
        all_features = wild_features+mut_features
        # print(all_features.shape)
        
        x = self.classifier(all_features)
        pred_cls = self.sigmoid(x)
        return pred_cls
    

class Classification(object):
    def __init__(self):
        super().__init__()
        self.model = Net(freeze_pretrained_bert=True, drop_prob=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.01)
            
    def data_setup(self, data_path="data/bpe_tokenized/train.full", batch_size=1):
        data =  pd.read_csv(data_path, header=None)
        print(data.shape)

        split_num=int(len(data) / batch_size)
        batched_data=np.array_split(data, split_num)
        print("Total batches: " + str(len(batched_data)))
        return batched_data
    
    def run_epoch(self, batch_df):
        wild_col, mut_col, label_col=0,1,2
        losses = []
        for tokens in batch_df.itertuples(index=False):
            # print(tokens[wild_col], tokens[mut_col], tokens[label_col])
            target_cls = 1 if tokens[label_col]=="stabilizing" else 0
            
            pred_cls = self.model(tokens[wild_col], tokens[mut_col])
            pred_cls = pred_cls.unsqueeze(dim=0)
            
            # print("here", pred_cls, torch.tensor([target_cls], dtype=torch.long))    
            loss = self.criterion(pred_cls, torch.tensor([target_cls], dtype=torch.long))
            losses.append(loss)
            # print(loss)
        epoch_loss = torch.stack(losses).mean()
        return epoch_loss
        
    def train(self):
        batched_data = self.data_setup(data_path="data/bpe_tokenized/train.full", batch_size=2)
        
        for count, batch_df in enumerate(batched_data):
            print(batch_df.shape)
            self.model.train()
            self.model.zero_grad()
            epoch_loss=self.run_epoch(batch_df)  
            epoch_loss.backward()
            print(epoch_loss)
            self.optimizer.step() 
             
            break
                
    
    
task = Classification()
task.train()    