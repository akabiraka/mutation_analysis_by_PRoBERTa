# Mutation stability analysis using PRoBERTa

#### Data preparation

|    Name    | File path                     | #Proteins | #Stabilizing | #Destabilizing |
| :--------: | ----------------------------- | :-------: | :----------: | :------------: |
|   Train   | data/dataset_5_train.csv      |    183    |     894     |      3119      |
| Validation | data/dataset_5_validation.csv |    20    |      53      |      277      |
|    Test    | data/dataset_5_test.csv       |    37    |      71      |      183      |

* The dataset is primarily collected from DeepDDG(https://pubs.acs.org/doi/10.1021/acs.jcim.8b00697) paper. The test set is the same as they proposed. A subset of training examples is randomly choosen from train as validation set. There is no overlapping proteins in train/validation/test set.
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

#### Training and testing (1)

* This uses the fairseq finetuning method to train the classifier. It is trained by the unbalanced dataset and it does not provide independence to design final classification layer. Therefore the report did not include them. See **Training and testing (2).**
* To finetune using the pretrained PRoBERTa model:

  * `bash models/mutation_classification_finetune.sh`
* To evaluate the model:

  * `python models/mutation_classification_eval.py`
* Conclusion: Since the dataset is biased towards one class (destabilizing), the model is highly biased.

#### Training and testing (2)

* This uses the PRoBERTa final layer features and trained a classifier layer. The report uses this approach since it samples balanced dataset for each batch. It also provides independence to design classification module.
* To train and test Model-0:
  * Train: `python MutationClassification0/train.py`
  * Test: `python MutationClassification0/test.py`
* To train and test Model-1:
  * To train: `python MutationClassification1/train.py`
  * To test: `python MutationClassification1/test.py`

#### Analysis

* To print class distribution: `python analyzers/data.py`
* To draw vocabulary embedding: `python analyzers/vocab_embedding.py`
* To draw sequence embedding: `python analyzers/seq_embedding.py`
* To draw loss values of training and validation epochs: `python analyzers/plot_losses.py`
