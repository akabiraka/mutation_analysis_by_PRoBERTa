# binarize the preprocessing data from uniprotkb-swissprot to create the dictionary
# fairseq-preprocess \
# 	--only-source \
#     --trainpref data/pretraining_data_tokenized.txt \
#     --destdir data/binarized/pretraining \

# binarize from seqs
fairseq-preprocess \
    --only-source \
    --trainpref data/tokenized/train.from \
    --validpref data/tokenized/val.from \
    --testpref data/tokenized/test.from \
    --destdir data/binarized/train/input0 \
    --srcdict data/binarized/pretraining/dict.txt

# binarize to seqs
fairseq-preprocess \
    --only-source \
    --trainpref data/tokenized/train.to \
    --validpref data/tokenized/val.to \
    --testpref data/tokenized/test.to \
    --destdir data/binarized/train/input1 \
    --srcdict data/binarized/pretraining/dict.txt

# binarize labels
fairseq-preprocess \
    --only-source \
    --trainpref data/tokenized/train.label \
    --validpref data/tokenized/val.label \
    --testpref data/tokenized/test.label \
    --destdir data/binarized/train/label

