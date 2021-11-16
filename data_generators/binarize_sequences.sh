# binarize the preprocessing data from uniprotkb-swissprot to create the dictionary
# fairseq-preprocess \
# 	--only-source \
#     --trainpref data/pretraining_data_tokenized.txt \
#     --destdir data/binarized/pretraining \

# binarize from seqs
fairseq-preprocess \
    --only-source \
    --trainpref data/tokenized/train.from \
    --destdir data/binarized/train/input0 \
    --srcdict data/binarized/pretraining/dict.txt

# binarize to seqs
fairseq-preprocess \
    --only-source \
    --trainpref data/tokenized/train.to \
    --destdir data/binarized/train/input1 \
    --srcdict data/binarized/pretraining/dict.txt

# binarize labels
fairseq-preprocess \
    --only-source \
    --trainpref data/tokenized/train.label \
    --destdir data/binarized/train/label \
