import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import torch
import pandas as pd
import models.MutDataset as MutDataset
import models.MutClassifier as MutClassifier
from torch.utils.data import DataLoader


fold_no=1
init_lr=0.0001
batch_size=16
epochs=1
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

# data specific things
wild_col, mut_col, label_col, fold_col=0,1,2,3
df = pd.read_csv("data/splits/all.cv", header=None)
class_dict = {label:i for i, label in enumerate(df[label_col].unique())}
n_classes = len(class_dict)

# dataset and loader
train_df, val_df = df[(df[fold_col]!=fold_no)], df[(df[fold_col]==fold_no)]
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)


model = MutClassifier.Pooler(n_classes).to(device)
# print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.01)

for epoch in range(1, epochs+1):
    train_batch_loader = MutDataset.get_batched_data(train_df, class_dict, batch_size)
    val_batch_loader = [val_df]
    train_loss = MutClassifier.train(model, train_batch_loader, class_dict, device, criterion, optimizer)
    val_loss = MutClassifier.test(model, train_batch_loader, class_dict, device, criterion)