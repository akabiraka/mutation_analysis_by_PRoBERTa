import sys
sys.path.append("../mutation_analysis_by_PRoBERTa")

import numpy as np
import pandas as pd

def get_batched_data(df, class_dict, batch_size):
    wild_col, mut_col, label_col = 0, 1, 2
    sample_size = int(batch_size/len(class_dict))
    n_batchs = int(len(df)/batch_size) # getting large enough number of batches will ensure that each datam is sampled at least once

    batched_data = []
    for i in range(1, n_batchs+1):
        a_batch = []
        for (key, value) in class_dict.items():
            sampled = df[df[label_col]==key].sample(n=sample_size)
            a_batch.append(sampled)
            
        batch = pd.concat(a_batch)
        shuffled = batch.sample(frac=1).reset_index(drop=True)
        
        # printing each batch class distribution
        # a = shuffled[(shuffled[label_col]=="destabilizing")].shape
        # b = shuffled[(shuffled[label_col]=="stabilizing")].shape
        # c = shuffled[(shuffled[label_col]=="neutral")].shape
        # print(f"Batch no: {i} ---> destabilizing: {a}, stabilizing: {b}, neutral: {c}")
    
        batched_data.append(shuffled)
    print(f"Total batches: {len(batched_data)}")   
    return batched_data

def get_batched_data_x(data, batch_size):
    """data is a df"""
    wild_col, mut_col, label_col = 0, 1, 2
    # data =  pd.read_csv(data_path, header=None)

    stabilizing = data[data[label_col]=="stabilizing"]
    destabilizing = data[data[label_col]=="destabilizing"]
    stab_n_rows, destab_n_rows = stabilizing.shape[0], destabilizing.shape[0]
    print(f"n_stabilizing: {stab_n_rows}, n_destabilizing: {destab_n_rows}")

    sample_size=int(batch_size/2)
    batched_data = []
    while destab_n_rows > 0:
        restart_stab_sampling_flag=False
        # if stabilizing rows < sample_size, select downsized batch
        if stab_n_rows<sample_size: 
            sample_size=stab_n_rows 
            restart_stab_sampling_flag=True
        # if destabilizing rows < sample_size, select downsized batch
        elif destab_n_rows<sample_size:
            sample_size=destab_n_rows
        else: sample_size=int(batch_size/2)
        
        # random sampling from stabilizing and destabilizing
        stab_sampled = stabilizing.sample(n=sample_size)
        destab_sampled = destabilizing.sample(n=sample_size)

        # shuffle the sampled data
        sampled = pd.concat([stab_sampled, destab_sampled])
        shuffled = sampled.sample(frac=1).reset_index(drop=True)
        
        # without replacement: remove the sampled rows 
        stabilizing = stabilizing.drop(stab_sampled.index)
        destabilizing = destabilizing.drop(destab_sampled.index)

        batched_data.append(shuffled)
        stab_n_rows, destab_n_rows = stabilizing.shape[0], destabilizing.shape[0]
        
        # if there is not stabilizing mutations, restart 
        if restart_stab_sampling_flag: 
            stabilizing = data[data[label_col]=="stabilizing"]
            stab_n_rows, destab_n_rows = stabilizing.shape[0], destabilizing.shape[0]
            # break
        
    print("Total batches: " + str(len(batched_data)))   
    # for i, batch_df in enumerate(batched_data):
    #     print(batch_df.shape)
    return batched_data