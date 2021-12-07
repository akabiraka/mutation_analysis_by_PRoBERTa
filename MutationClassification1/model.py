import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, drop_prob=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.Dropout(p=drop_prob),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(p=drop_prob),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Dropout(p=drop_prob)
        )
        self.softmax = nn.Softmax()
        
    
    def forward(self, wild_features, mut_features):
        # all_features = wild_features+mut_features
        # print(wild_features.shape)
        all_features = torch.cat((wild_features, mut_features), dim=0)
        # print(all_features.shape)
        
        x = self.classifier(all_features)
        pred_cls = self.softmax(x)
        return pred_cls
    
