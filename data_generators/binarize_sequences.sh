# binarize the preprocessing data from uniprotkb-swissprot to create the dictionary
# fairseq-preprocess \
# 	--only-source \
#     --trainpref data/pretraining_data_tokenized.txt \
#     --destdir data/binarized/pretraining \

# binarize from seqs
fairseq-preprocess \
    --only-source \
    --trainpref data/bpe_tokenized/train.from \
    --validpref data/bpe_tokenized/val.from \
    --testpref data/bpe_tokenized/test.from \
    --destdir data/bpe_binarized/mutation_classification/input0 \
    --srcdict data/pretrained_models/dict.txt 

# binarize to seqs
fairseq-preprocess \
    --only-source \
    --trainpref data/bpe_tokenized/train.to \
    --validpref data/bpe_tokenized/val.to \
    --testpref data/bpe_tokenized/test.to \
    --destdir data/bpe_binarized/mutation_classification/input1 \
    --srcdict data/pretrained_models/dict.txt 

# binarize labels
fairseq-preprocess \
    --only-source \
    --trainpref data/bpe_tokenized/train.label \
    --validpref data/bpe_tokenized/val.label \
    --testpref data/bpe_tokenized/test.label \
    --destdir data/bpe_binarized/mutation_classification/label

