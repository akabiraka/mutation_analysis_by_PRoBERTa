import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import torch
import torch.nn as nn
import pandas as pd

from MutationClassification.model import Net
from fairseq.models.roberta import RobertaModel



def print_metrics(target_classes, pred_classes):
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, log_loss
    print("confusion_matrix: ", confusion_matrix(target_classes, pred_classes))
    print("accuracy: ", accuracy_score(target_classes, pred_classes))
    print("recall: ", recall_score(target_classes, pred_classes))
    print("precision: ", precision_score(target_classes, pred_classes))
    print("f1_score: ", f1_score(target_classes, pred_classes))
    fpr, tpr, treshold = roc_curve(target_classes, pred_classes)
    print("fpr, tpr, th: ", fpr, tpr, treshold)
    roc_auc = auc(fpr, tpr)
    print("roc_auc: ", roc_auc)
    print("log_loss: ", log_loss(target_classes, pred_classes))

def test(test_data_path, model):
    batch_df =  pd.read_csv(test_data_path, header=None)
    wild_col, mut_col, label_col=0,1,2
    losses = []
    target_classes, pred_classes=[], []
    for tokens in batch_df.itertuples(index=False):
        target_cls = 1 if tokens[label_col]=="stabilizing" else 0
        target_cls = torch.tensor([target_cls], dtype=torch.long).to(device)
        
        wild_encoded = roberta_model.encode(tokens[wild_col])
        mut_encoded = roberta_model.encode(tokens[mut_col])
        
        if len(wild_encoded)>512 or len(mut_encoded)>512:
            print("tokens exceeds maximum length: {}, {} > 512".format(len(wild_encoded), len(mut_encoded)))
            continue
        wild_features = roberta_model.extract_features(wild_encoded).sum(dim=1).squeeze().to(device)
        mut_features = roberta_model.extract_features(mut_encoded).sum(dim=1).squeeze().to(device)
        # print(wild_features.shape, mut_features.shape)
        
        pred_cls = model(wild_features, mut_features)
        pred_cls = pred_cls.unsqueeze(dim=0)
        # print(pred_cls)
        
        per_item_loss = criterion(pred_cls, target_cls)
        losses.append(per_item_loss)
        
        pred_classes.append(pred_cls.argmax().item())
        target_classes.append(target_cls[0].item())
        # break
        
    batch_loss = torch.stack(losses).mean()
    print("Cross entropy loss: ", batch_loss)
    print_metrics(target_classes, pred_classes)
        


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

roberta_model = RobertaModel.from_pretrained(model_name_or_path="data/pretrained_models/", checkpoint_file="checkpoint_best.pt", 
                                                     bpe="sentencepiece", sentencepiece_model="data/bpe_model/m_reviewed.model")
 
roberta_model.to(device)
roberta_model.eval()

criterion = nn.CrossEntropyLoss()

test_data_path="data/bpe_tokenized/test.full"

for i in range(1, 13):
    print("Model: ", i)
    model = Net(drop_prob=0.5).to(device)
    model.load_state_dict(torch.load("outputs/models_mut_classify_balanced_data/{}_mut_classify.pt".format(i), map_location=torch.device(device)))
    model.eval()

    test(test_data_path, model)

