# binarize the preprocessing data from uniprotkb-swissprot to create the dictionary
# fairseq-preprocess \
# 	--only-source \
#     --trainpref data/pretraining_data_tokenized.txt \
#     --destdir data/binarized/pretraining \


fairseq-preprocess \
    --only-source \
    --trainpref data/tokenized/train.sequence \
    --destdir data/binarized/mutation \
    --srcdict data/binarized/pretraining/dict.txt
