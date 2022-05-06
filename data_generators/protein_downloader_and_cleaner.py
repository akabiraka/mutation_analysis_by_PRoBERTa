import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import pandas as pd
from objects.PDBData import PDBData
from objects.Selector import ChainAndAminoAcidSelect

# configurations
pdb_dir = "data/pdbs/"
pdbs_clean_dir = "data/pdbs_clean/"
fastas_dir = "data/fastas/"
CIF = "mmCif"

# input_file_path = "data/dataset_5_train.csv"
input_file_path = "data/dataset_5_test.csv"
n_rows_to_skip = 0
n_rows_to_evalutate = 100000

# object initialization
pdbdata = PDBData(pdb_dir=pdb_dir)

df = pd.read_csv(input_file_path)

for i, row in df.iterrows():
    if i+1 <= n_rows_to_skip: continue
    
    # extracting the data
    pdb_id = row["pdb_id"].lower()[:4]
    chain_id = row["chain_id"]
    mutation = row["mutation"]
    mutation_site = int(row["mutation_site"])
    wild_residue = row["wild_residue"]
    mutant_residue = row["mutant_residue"]
    ddg = row["ddG"]
    
    # creating necessary file paths
    cln_pdb_file = pdbs_clean_dir+pdb_id+chain_id+".pdb"
    wild_fasta_file = fastas_dir+pdb_id+chain_id+".fasta"
    mutant_fasta_file = fastas_dir+pdb_id+chain_id+"_"+mutation+".fasta"

    # download and clean PDB, generate fasta
    pdbdata.download_structure(pdb_id=pdb_id)
    pdbdata.clean(pdb_id=pdb_id, chain_id=chain_id, selector=ChainAndAminoAcidSelect(chain_id))
    pdbdata.generate_fasta_from_pdb(pdb_id=pdb_id, chain_id=chain_id, input_pdb_filepath=cln_pdb_file, save_as_fasta=True, output_fasta_dir=fastas_dir)
    
    # generate mutant fasta
    zero_based_mutation_site = pdbdata.get_zero_based_mutation_site(cln_pdb_file, chain_id, mutation_residue_id=(" ", mutation_site, " "))
    pdbdata.create_mutant_fasta_file(wild_fasta_file, mutant_fasta_file, zero_based_mutation_site, mutant_residue, mutation)
    print("Row no:{}->{}{}, mutation:{}, 0-based_mutaiton_site:{}".format(i+1, pdb_id, chain_id, mutation, zero_based_mutation_site))
    print()
    if i+1 == n_rows_to_skip+n_rows_to_evalutate: 
        break
