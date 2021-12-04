from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
import pandas as pd
import numpy as np
import torch

batch_size = 1
from_col=0
to_col=1
label_col=2
classification_head="mutation_classification"
model_dir = "outputs/mutation_classification/checkpoints/"
binarized_path="data/bpe_binarized/mutation_classification/"

data =  pd.read_csv("data/bpe_tokenized/test.full", header=None)
print(data.shape)

split_num=int(len(data) / batch_size)
batched_data=np.array_split(data, split_num)
print("Total batches: " + str(len(batched_data)))

#roberta_model = RobertaModel.from_pretrained(model_name_or_path=model_dir, checkpoint_file="checkpoint_best.pt", data_name_or_path=binarized_path)
roberta_model = RobertaModel.from_pretrained(model_name_or_path=model_dir,
checkpoint_file="checkpoint_best.pt",  data_name_or_path=binarized_path, bpe="sentencepiece", sentencepiece_model="data/bpe_model/m_reviewed.model")
roberta_model.eval()

label_fn = lambda label: roberta_model.task.label_dictionary.string(
    [label + roberta_model.task.label_dictionary.nspecial]
)

for count, batch_df in enumerate(batched_data):
    for tokens in batch_df.itertuples(index=False): 
        print("inputs: ", tokens[from_col], tokens[to_col])
        encoded = roberta_model.encode(tokens[from_col], tokens[to_col])
        print("encoded: ", encoded)
        last_layer_features = roberta_model.extract_features(encoded)
        print("last_layer_feature_size: ", last_layer_features.size())
        decoded = roberta_model.decode(encoded)
        print("decoded: ", decoded)
        batch=collate_tokens([torch.cat((roberta_model.encode(tokens[from_col], tokens[to_col]), torch.ones(512, dtype = torch.long)))[:512]
            for tokens in batch_df.itertuples(index=False)], pad_idx=1)
        # print(batch)
        logprobs = label_fn(roberta_model.predict(classification_head, batch).argmax().item())      
        print("target: ", tokens[label_col], " pred: ", logprobs)
   # break

# tokens = model.encode('▁MI VF VR FN SS')
# print(tokens)
# out = model.decode(tokens)
# print(out)
# tokens = ["▁MAS", "PT", "VKL"]
# model.encode(tokens)
# , "data/binarized/mutation"
