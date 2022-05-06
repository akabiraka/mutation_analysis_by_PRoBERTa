import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import torch
from fairseq.models.roberta import RobertaModel

class Pooler(torch.nn.Module):
    def __init__(self, n_classes, inner_dim=768, drop_prob=0.3) -> None:
        super(Pooler, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name_or_path="data/pretrained_models/", 
                                                          checkpoint_file="checkpoint_best.pt",
                                                          bpe="sentencepiece", 
                                                          sentencepiece_model="data/bpe_model/m_reviewed.model")
        self.classifier = torch.nn.Sequential(torch.nn.Linear(2*768, inner_dim),
                                              torch.nn.Dropout(p=drop_prob),
                                              torch.nn.LeakyReLU(),
                                              torch.nn.Linear(inner_dim, n_classes))


    def forward(self, seq1, seq2):
        # tokens = self.roberta.encode(seq1+","+seq2)
        # features = self.roberta.extract_features(tokens)
        # print(tokens)
        # print(features.shape)

        tokens1 = self.roberta.encode(seq1)
        tokens2 = self.roberta.encode(seq2)
        # print(tokens1, tokens2)

        features1 = self.roberta.extract_features(tokens1) # Extract the last layer's features, [batch_size, tokens_len, 786]
        features2 = self.roberta.extract_features(tokens2) # Extract the last layer's features, [batch_size, tokens_len, 786]
        # print(features1.shape, features2.shape)
        
        x1 = features1[:, 0, :]  # take <CLS> token, [batch_size, 786]
        x2 = features2[:, 0, :]  # take <CLS> token, [batch_size, 786]
        # print(x1.shape, x2.shape)
        
        x = torch.cat((x1, x2), dim=1) # [batch_size, 2*786]
        # print(x.shape) 
        out = self.classifier(x)
        # print(out.shape) # [batch_size, n_classes]
        return out


# example usage
# seq1 = "▁M KVI FLKD VKG KG KKG EIKN VADG YAN NFL FKQ GLAI EAT P ANL KAL EAQ K"
# seq2 = "▁M KVI FLKD VKG KG KKG EE KN VADG YAN NFL FKQ GLAI EAT P ANL KAL EAQ K"

# m = Pooler(2)
# print(m)
# out = m(seq1, seq2)
# print(out)

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

def run_batch(model, batch_df, device, criterion, class_dict, print_metrics=False):
        wild_col, mut_col, label_col=0,1,2
        losses = []
        target_classes, pred_classes=[], []
        for tokens in batch_df.itertuples(index=False):
            # print(tokens[wild_col], tokens[mut_col], tokens[label_col])
            target_cls = class_dict[tokens[label_col]]
            target_cls = torch.tensor([target_cls], dtype=torch.long).to(device)

            pred_cls = model(tokens[wild_col], tokens[mut_col])
            # pred_cls = pred_cls.unsqueeze(dim=0)
            # print(pred_cls)
            
            per_item_loss = criterion(pred_cls, target_cls)
            losses.append(per_item_loss)
            
            pred_classes.append(pred_cls.argmax().item())
            target_classes.append(target_cls[0].item())
            # break
        batch_loss = torch.stack(losses).mean()
        if print_metrics: print_metrics(target_classes, pred_classes)
        return batch_loss

def train(model, train_batch_loader, class_dict, device, criterion, optimizer):
    model.train()
    losses = []
    for batch_no, batch_df in enumerate(train_batch_loader):
        model.zero_grad()
        batch_loss=run_batch(model, batch_df, device, criterion, class_dict)  
        batch_loss.backward()
        optimizer.step() 
        losses.append(batch_loss)
        print("train batch_no:{}, batch_shape:{}, batch_loss:{}".format(batch_no, batch_df.shape, batch_loss))
        # break
    epoch_loss = torch.stack(losses).mean().item()
    return epoch_loss

@torch.no_grad()
def test(model, batch_loader, class_dict, device, criterion):
    model.eval()
    losses = []
    for batch_no, batch_df in enumerate(batch_loader):
        batch_loss=run_batch(model, batch_df, device, criterion, class_dict, print_metrics=True)
        losses.append(batch_loss)
        print("test batch_no:{}, batch_shape:{}, batch_loss:{}".format(batch_no, batch_df.shape, batch_loss))
        # break
    epoch_loss = torch.stack(losses).mean().item()
    return epoch_loss