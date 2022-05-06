import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import pandas as pd
from sklearn.model_selection import StratifiedKFold



data_path="data/bpe_tokenized/train.full"
wild_col, mut_col, label_col, fold_col=0,1,2,3
df = pd.read_csv(data_path, header=None)
# print(df)

df = df.sample(frac=1).reset_index(drop=True)
skf = StratifiedKFold(n_splits = 5)

for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df[label_col].values)):
    df.loc[val_idx, fold_col] = str(fold)

# print(df)

df.to_csv("data/splits/all.cv", index=False, header=None)

print("Data distribution in each fold: ")
for fold in df[fold_col].unique():
    a = df[(df[fold_col]==fold) & (df[label_col]=="destabilizing")].shape
    b = df[(df[fold_col]==fold) & (df[label_col]=="stabilizing")].shape
    c = df[(df[fold_col]==fold) & (df[label_col]=="neutral")].shape
    print(f"Fold: {fold} ---> destabilizing: {a}, stabilizing: {b}, neutral: {c}")