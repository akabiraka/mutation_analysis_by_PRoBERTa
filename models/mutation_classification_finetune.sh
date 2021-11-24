#!/usr/bin/sh

#SBATCH --job-name=mutation_classification
#SBATCH --qos=csqos
#SBATCH --output=/scratch/akabir4/mutation_analysis_by_PRoBERTa/outputs/argo_logs/mutation_classification-%N-%j.output
#SBATCH --error=/scratch/akabir4/mutation_analysis_by_PRoBERTa/outputs/argo_logs/mutation_classification-%N-%j.error
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --mem=16000MB

JOB="mutation_classification"
DATA_DIR=data/binarized/$JOB
OUTPUT_DIR=outputs/$JOB
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p $CHECKPOINT_DIR

NUM_GPUS=1

ENCODER_EMBED_DIM=768
ENCODER_LAYERS=5
TOTAL_UPDATES=12500
WARMUP_UPDATES=3125
PEAK_LR=0.0025
MAX_SENTENCES=32
UPDATE_FREQ=64

NUM_CLASSES=2
PATIENCE=5

TOKENS_PER_SAMPLE=512
MAX_POSITIONS=512

PREFIX=$JOB
PREFIX="$PREFIX.DIM_$ENCODER_EMBED_DIM.LAYERS_$ENCODER_LAYERS"
PREFIX="$PREFIX.UPDATES_$TOTAL_UPDATES.WARMUP_$WARMUP_UPDATES"
PREFIX="$PREFIX.LR_$PEAK_LR.BATCH_$MAX_SENTENCES.PATIENCE_$PATIENCE"
LOG_FILE="$OUTPUT_DIR/$PREFIX.log"

ROBERTA_PATH=data/pretrained_models/checkpoint_best.pt
CLASSIFICATION_HEAD=$JOB

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --max-positions $MAX_POSITIONS --batch-size $MAX_SENTENCES  \
    --arch roberta_base --task sentence_prediction \
    --classification-head-name $CLASSIFICATION_HEAD \
    --restore-file $ROBERTA_PATH --reset-optimizer --reset-dataloader --reset-meters \
    --init-token 0 --separator-token 2 \
    --criterion sentence_prediction --num-classes $NUM_CLASSES \
    --optimizer adam \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES \
    --encoder-embed-dim $ENCODER_EMBED_DIM --encoder-layers $ENCODER_LAYERS \
    --save-dir "$CHECKPOINT_DIR" --save-interval 1 --save-interval-updates 100 --keep-interval-updates 5 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --patience $PATIENCE \
    --log-format simple --log-interval 1000 2>&1 | tee -a $LOG_FILE
