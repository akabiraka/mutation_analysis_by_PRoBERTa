#!/usr/bin/sh

## this must be run from directory where run.py exists.
## --workdir is not used in this file.

#SBATCH --job-name=general
#SBATCH --qos=csqos
#SBATCH --output=/scratch/akabir4/mutation_analysis_by_PRoBERTa/outputs/argo_logs/general-%N-%j.output
#SBATCH --error=/scratch/akabir4/mutation_analysis_by_PRoBERTa/outputs/argo_logs/general-%N-%j.error
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=all-HiPri
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000MB

##python analyzers/vocab_embedding.py 
python analyzers/protein_seq_embedding.py
