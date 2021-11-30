# Mutation stability analysis using PRoBERTa



#### Data preparation

| Name       | File path                     | #Proteins | #Stabilizing | #Destabilizing |
| ---------- | ----------------------------- | --------- | ------------ | -------------- |
| Train      | data/dataset_5_train.csv      |           |              |                |
| Validation | data/dataset_5_validation.csv |           |              |                |
| Test       | data/dataset_5_test.csv       |           |              |                |

* The dataset is primarily collected from DeepDDG(https://pubs.acs.org/doi/10.1021/acs.jcim.8b00697) paper. The test set is the same as they proposed. A subset of training examples is randomly choosen from train as validation set. There is not overlapping proteins in train/validation/test set.
* To download, clean and generate fasta sequence:
  * `python data_generators/protein_downloader_and_cleaner.py`
* Tokenize sequences:
  * To train tokenizer:
    * `from data_generators.tokenizer import *`
    * `train_tokenizer()`
  * To tokenize sequences using pretrained model:
    * `python data_generators/tokenizer.py`
* Binarize sequences:
  * `bash data_generators/binarize_sequences.sh`

#### Download pretrained models

All the following links corresponds to PRoBERTa:

* Paper: Transforming the Language of Life: Transformer Neural Networks for Protein Prediction Tasks

* Github link: https://github.com/annambiar/PRoBERTa

Download the following pretrained models from PRoBERTa:

* BPE model: https://drive.google.com/drive/folders/1lJkG4IAWxSs8mGqSk-MjsaBQFV4Y3dhq?usp=sharing
* PRoBERTa model (checkpoint_best.pt): https://drive.google.com/file/d/1IZFE71DgFy0RDKYsx0I28PnRlDXNUP_T/view?usp=sharing

#### Training and testing

* To finetune using the pretrained PRoBERTa model:
  * `bash models/mutation_classification_finetune.sh`
* To evaluate the model:
  * `python models/mutation_classification_eval.py`
* Conclusion: Since the dataset is biased towards one class (destabilizing), the model is highly biased.

#### Training and testing

* To train:
  * `python MutationClassification/model.py`
  * This `MutationClassification/model.py` contains:
    * `Net`: This is the neural network class.
    * `Classification`: This does random sampling without replacement in function `get_batched_data`, `train` and `validate` the model. The `run` function does all the works.
  *
