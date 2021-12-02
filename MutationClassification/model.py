import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

class Net(nn.Module):
    def __init__(self, device, freeze_pretrained_bert=True, drop_prob=0.5):
        super().__init__()
        self.device=device
        self.roberta_model = RobertaModel.from_pretrained(model_name_or_path="data/pretrained_models/", checkpoint_file="checkpoint_best.pt", 
                                                     bpe="sentencepiece", sentencepiece_model="data/bpe_model/m_reviewed.model")
        self.roberta_model.to(self.device)
        if freeze_pretrained_bert: self.roberta_model.eval()
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 2),
            nn.Dropout(p=drop_prob)
        )
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, wild_tokens, mut_tokens):
        wild_encoded = self.roberta_model.encode(wild_tokens)
        mut_encoded = self.roberta_model.encode(mut_tokens)
        
        wild_features = self.roberta_model.extract_features(wild_encoded).sum(dim=1).squeeze().to(self.device)
        mut_features = self.roberta_model.extract_features(mut_encoded).sum(dim=1).squeeze().to(self.device)
        # print(wild_features.shape, mut_features.shape)
        
        all_features = wild_features+mut_features
        # print(all_features.shape)
        
        x = self.classifier(all_features)
        pred_cls = self.sigmoid(x)
        return pred_cls
    

class Classification(object):
    def __init__(self, init_lr, n_epochs, batch_size):
        super().__init__()
        self.init_lr=init_lr
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Net(self.device, freeze_pretrained_bert=True, drop_prob=0.5).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.init_lr, weight_decay=0.01)
        
            
    def data_setup(self, data_path):
        data =  pd.read_csv(data_path, header=None)
        print(data.shape)

        split_num=int(len(data) / self.batch_size)
        batched_data=np.array_split(data, split_num)
        print("Total batches: " + str(len(batched_data)))
        return batched_data
    
    
    def get_batched_data(self, data_path):
        wild_col, mut_col, label_col = 0, 1, 2
        data =  pd.read_csv(data_path, header=None)

        stabilizing = data[data[label_col]=="stabilizing"]
        destabilizing = data[data[label_col]=="destabilizing"]
        stab_n_rows, destab_n_rows = stabilizing.shape[0], destabilizing.shape[0]
        print(stab_n_rows, destab_n_rows)

        sample_size=int(self.batch_size/2)
        batched_data = []
        while destab_n_rows > 0:
            restart_stab_sampling_flag=False
            # if stabilizing rows < sample_size, select downsized batch
            if stab_n_rows<sample_size: 
                sample_size=stab_n_rows 
                restart_stab_sampling_flag=True
            # if destabilizing rows < sample_size, select downsized batch
            elif destab_n_rows<sample_size:
                sample_size=destab_n_rows
            else: sample_size=int(self.batch_size/2)
            
            # random sampling from stabilizing and destabilizing
            stab_sampled = stabilizing.sample(n=sample_size)
            destab_sampled = destabilizing.sample(n=sample_size)

            # shuffle the sampled data
            sampled = pd.concat([stab_sampled, destab_sampled])
            shuffled = sampled.sample(frac=1).reset_index(drop=True)
            
            # without replacement: remove the sampled rows 
            stabilizing = stabilizing.drop(stab_sampled.index)
            destabilizing = destabilizing.drop(destab_sampled.index)

            batched_data.append(shuffled)
            stab_n_rows, destab_n_rows = stabilizing.shape[0], destabilizing.shape[0]
            
            # if there is not stabilizing mutations, restart 
            if restart_stab_sampling_flag: 
                stabilizing = data[data[label_col]=="stabilizing"]
                stab_n_rows, destab_n_rows = stabilizing.shape[0], destabilizing.shape[0]
                # break
            
        print("Total batches: " + str(len(batched_data)))   
        # for i, batch_df in enumerate(batched_data):
        #     print(batch_df.shape)
        return batched_data
    
    
    def run_batch(self, batch_df):
        wild_col, mut_col, label_col=0,1,2
        losses = []
        for tokens in batch_df.itertuples(index=False):
            # print(tokens[wild_col], tokens[mut_col], tokens[label_col])
            target_cls = 1 if tokens[label_col]=="stabilizing" else 0
            target_cls = torch.tensor([target_cls], dtype=torch.long).to(self.device)
            
            pred_cls = self.model(tokens[wild_col], tokens[mut_col])
            pred_cls = pred_cls.unsqueeze(dim=0)
            # print(pred_cls)
            
            per_item_loss = self.criterion(pred_cls, target_cls)
            losses.append(per_item_loss)
            # break
        batch_loss = torch.stack(losses).mean()
        return batch_loss
    
    
    def train(self, train_data_path):
        batched_data = self.get_batched_data(train_data_path)
        self.model.train()
        losses = []
        for batch_no, batch_df in enumerate(batched_data):
            self.model.zero_grad()
            batch_loss=self.run_batch(batch_df)  
            batch_loss.backward()
            self.optimizer.step() 
            losses.append(batch_loss)
            print("batch_no:{}, batch_shape:{}, batch_loss:{}".format(batch_no, batch_df.shape, batch_loss))
            #break
        epoch_loss = torch.stack(losses).mean().item()
        return epoch_loss


    def validate(self, val_data_path):
        self.model.eval()
        losses=[]
        data =  pd.read_csv(val_data_path, header=None)
        val_loss = self.run_batch(data)
        return val_loss.item()
        
    def run(self, train_data_path, val_data_path):
        best_loss = np.inf        
        train_losses = []
        val_losses = []
        model_path="outputs/models/mut_classify_lr_{}_epoch_{}_batch_{}.pt".format(self.init_lr, self.n_epochs, self.batch_size)

        for epoch in range(1, self.n_epochs+1):
            train_loss = self.train(train_data_path)
            val_loss = self.validate(val_data_path)
            print("[{}/{}] train_loss:{:.4f}, val_loss:{:.4f}".format(epoch, self.n_epochs, train_loss, val_loss))
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_loss:
                torch.save(self.model.state_dict(), model_path) 
            #break
        print("train_losses: ", train_losses)
        print("val_losses: ", val_losses)
                
    
train_data_path="data/bpe_tokenized/train.full"    
val_data_path="data/bpe_tokenized/val.full"    
task = Classification(init_lr=0.00001, batch_size=32,  n_epochs=50)
task.run(train_data_path, val_data_path)    

