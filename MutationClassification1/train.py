import sys

from torch._C import device
sys.path.append("../mutation_analysis_by_PRoBERTa")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from MutationClassification1.model import *
from MutationClassification1.dataset import get_batched_data
from fairseq.models.roberta import RobertaModel

class Classification(object):
    def __init__(self, init_lr, batch_size, n_epochs, criterion, device):
        super().__init__()
        self.init_lr=init_lr
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.device = device

        self.model = Net_2(drop_prob=0.5).to(self.device)
        self.roberta_model = self.init_roberta_model()
        
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.init_lr, weight_decay=0.01)
        
    def init_roberta_model(self):
        roberta_model = RobertaModel.from_pretrained(model_name_or_path="data/pretrained_models/", checkpoint_file="checkpoint_best.pt", 
                                                     bpe="sentencepiece", sentencepiece_model="data/bpe_model/m_reviewed.model")
        roberta_model.to(self.device)
        roberta_model.eval()
        return roberta_model
    
    def print_metrics(self, target_classes, pred_classes):
        from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, log_loss
        print("confusion_matrix: ", confusion_matrix(target_classes, pred_classes))
        # print("accuracy: ", accuracy_score(target_classes, pred_classes))
        # print("recall: ", recall_score(target_classes, pred_classes))
        # print("precision: ", precision_score(target_classes, pred_classes))
        # print("f1_score: ", f1_score(target_classes, pred_classes))
        # fpr, tpr, treshold = roc_curve(target_classes, pred_classes)
        # print("fpr, tpr, th: ", fpr, tpr, treshold)
        # roc_auc = auc(fpr, tpr)
        # print("roc_auc: ", roc_auc)
        # print("log_loss: ", log_loss(target_classes, pred_classes))
        
        
    def run_batch(self, batch_df):
        wild_col, mut_col, label_col=0,1,2
        losses = []
        target_classes, pred_classes=[], []
        for tokens in batch_df.itertuples(index=False):
            # print(tokens[wild_col], tokens[mut_col], tokens[label_col])
            target_cls = 1 if tokens[label_col]=="stabilizing" else 0
            target_cls = torch.tensor([target_cls], dtype=torch.float32).to(self.device)
            
            wild_encoded = self.roberta_model.encode(tokens[wild_col])
            mut_encoded = self.roberta_model.encode(tokens[mut_col])
            wild_features = self.roberta_model.extract_features(wild_encoded).sum(dim=1).squeeze().to(self.device)
            mut_features = self.roberta_model.extract_features(mut_encoded).sum(dim=1).squeeze().to(self.device)
            # print(wild_features.shape, mut_features.shape)
            
            pred_cls = self.model(wild_features, mut_features)
            # pred_cls = pred_cls.unsqueeze(dim=0)
            # print(pred_cls)
            
            per_item_loss = self.criterion(pred_cls, target_cls)
            losses.append(per_item_loss)
            
            # print(pred_cls, target_cls, per_item_loss)

            pred_classes.append(1 if pred_cls[0].item()>0.5 else 0)
            target_classes.append(target_cls[0].item())
            # break
        batch_loss = torch.stack(losses).mean()
        self.print_metrics(target_classes, pred_classes)
        return batch_loss
    
    
    def train(self, train_data_path):
        batched_data = get_batched_data(train_data_path, self.batch_size)
        self.model.train()
        losses = []
        for batch_no, batch_df in enumerate(batched_data):
            self.model.zero_grad()
            batch_loss=self.run_batch(batch_df)  
            batch_loss.backward()
            self.optimizer.step() 
            losses.append(batch_loss.item()) # loss gradient will be accumulated unless taking the item
            print("batch_no:{}, batch_shape:{}, batch_loss:{}".format(batch_no, batch_df.shape, batch_loss.item()))
            # break
        epoch_loss = np.mean(losses)
        return epoch_loss


    def validate(self, val_data_path):
        with torch.no_grad():
            self.model.eval()
            losses=[]
            data =  pd.read_csv(val_data_path, header=None)
            val_loss = self.run_batch(data)
        return val_loss.item()
        
    def run(self, run_no, train_data_path, val_data_path):
        best_loss = np.inf        
        train_losses = []
        val_losses = []
        model_path="outputs/models_mut_classify_balanced_data/{}_mut_classify.pt".format(run_no)

        for epoch in range(1, self.n_epochs+1):
            print("training...")
            train_loss = self.train(train_data_path)
            print("validating...")
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

# param_grid={
#     "lr":[0.001, 0.0001],
#     "batch_size":[32, 64],
#     "criterion_weight":[torch.tensor([0.6, 0.4]), torch.tensor([0.5, 0.5]), torch.tensor([0.4, 0.6])]
# }
# for run_no, params in enumerate(list(ParameterGrid(param_grid)), 1):
#     print("run: ", run_no, params["lr"], params["batch_size"], params["criterion_weight"])

#     task = Classification(init_lr=params["lr"], batch_size=params["batch_size"], n_epochs=50, criterion_weight=params["criterion_weight"])
#     task.run(run_no, train_data_path, val_data_path)    

param_grid={
    "lr":[0.0001, 0.00001],
    "batch_size":[64, 128],
    "criterion_weight":[torch.tensor([0.5, 0.5])]
}
for run_no, params in enumerate(list(ParameterGrid(param_grid)), 16):
    print("run: ", run_no, params["lr"], params["batch_size"], params["criterion_weight"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # task = Classification(init_lr=params["lr"], batch_size=params["batch_size"], n_epochs=50, criterion=nn.CrossEntropyLoss(weight=params["criterion_weight"].to(device)), device=device)
    task = Classification(init_lr=params["lr"], batch_size=params["batch_size"], n_epochs=50, criterion=nn.MSELoss(), device=device)
    task.run(run_no, train_data_path, val_data_path)    


