from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
import pandas as pd
import numpy as np
import torch

batch_size = 5
from_col=0
to_col=1
label_col=2
mutation_classification_head="mutation_cls_prediction"
model_folder = "data/pretrained_models/"
pretrained_checkpoint = "checkpoint_best.pt"
dict_path = "data/binarized/train/from/"

data =  pd.read_csv("data/tokenized/train.full", header=None)
print(data.shape)

split_num=int(len(data) / batch_size)
batched_data=np.array_split(data, split_num)
print("Total batches: " + str(len(batched_data)))

roberta_model = RobertaModel.from_pretrained(model_folder, pretrained_checkpoint)
# model.eval()
roberta_model.register_classification_head(mutation_classification_head, num_classes=2)

label_fn = lambda label: roberta_model.task.label_dictionary.string(
    [label + roberta_model.task.label_dictionary.nspecial]
)

for count, batch_df in enumerate(batched_data):
    for tokens in batch_df.itertuples(index=False):
        print(tokens[from_col], tokens[to_col], tokens[label_col])
        # model.encode(tokens[from_col], tokens[to_col])
        # torch.ones(512, dtype = torch.long) 
        batch=collate_tokens([torch.cat((roberta_model.encode(tokens[from_col], tokens[to_col]), torch.ones(512, dtype = torch.long)))[:512]
            for tokens in batch_df.itertuples(index=False)], pad_idx=1)
        # print(batch)
        logprobs = label_fn(roberta_model.predict(mutation_classification_head, batch).argmax().item())      
        print(logprobs)
    break

# tokens = model.encode('▁MI VF VR FN SS')
# print(tokens)
# out = model.decode(tokens)
# print(out)
# tokens = ["▁MAS", "PT", "VKL"]
# model.encode(tokens)
# , "data/binarized/mutation"