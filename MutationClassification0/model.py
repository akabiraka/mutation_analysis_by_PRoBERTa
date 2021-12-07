import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import torch.nn as nn

class Net(nn.Module):
    def __init__(self, drop_prob=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 2),
            nn.Dropout(p=drop_prob)
        )
        self.softmax = nn.Softmax()
        
    
    def forward(self, wild_features, mut_features):
        all_features = wild_features+mut_features
        # print(all_features.shape)
        
        x = self.classifier(all_features)
        pred_cls = self.softmax(x)
        return pred_cls
    