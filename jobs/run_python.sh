#!/usr/bin/sh

## this must be run from directory where run.py exists.
## --workdir is not used in this file.

#SBATCH --job-name=mut_classify
#SBATCH --qos=csqos
#SBATCH --output=/scratch/akabir4/mutation_analysis_by_PRoBERTa/outputs/argo_logs/mut_classify-%N-%j.output
#SBATCH --error=/scratch/akabir4/mutation_analysis_by_PRoBERTa/outputs/argo_logs/mut_classify-%N-%j.error
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

##cpu jobs
##SBATCH --partition=all-HiPri
##SBATCH --cpus-per-task=4

##GPU jobs
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --mem=16000MB
##SBATCH --time=1-24:00

##python analyzers/vocab_embedding.py 
##python analyzers/protein_seq_embedding.py
python MutationClassification/train.py
