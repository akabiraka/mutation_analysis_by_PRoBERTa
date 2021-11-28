#!/usr/bin/sh

#SBATCH --job-name=vocab_embed
#SBATCH --qos=csqos
#SBATCH --output=/scratch/akabir4/mutation_analysis_by_PRoBERTa/outputs/argo_logs/vocab_embed-%N-%j.output
#SBATCH --error=/scratch/akabir4/mutation_analysis_by_PRoBERTa/outputs/argo_logs/vocab_embed-%N-%j.error
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=all-HiPri
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000MB

python analyzers/vocab_embedding.py 
