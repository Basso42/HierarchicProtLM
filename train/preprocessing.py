import polars as pl
import pandas as pd
import numpy as np

from settings import gen_dataset

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import sys
import os
import json

if gen_dataset=='ECPred40':
    print("Reading the ECPred40 .json files and rearranging train, validation and test from it")

    train, validation, test = pd.read_json(data_path + 'ECPred40_train.json').drop('index',axis=1).rename(columns={"sequence": "AA_seq"}), pd.read_json(data_path + 'ECPred40_valid.json').rename(columns={"sequence": "AA_seq"}), pd.read_json(data_path + 'ECPred40_test.json').rename(columns={"sequence": "AA_seq"})

    train_labels = sorted(set(train['EC Number'].tolist()))
    label2idx = {ec:i for i,ec in enumerate(train_labels)}
    train['label'], validation['label'], test['label'] = train['EC Number'].map(label2idx), validation['EC Number'].map(label2idx), test['EC Number'].map(label2idx)

    weights = compute_class_weight(class_weight='balanced', classes=np.sort(train['label'].unique()), y=train['label'])
    
    print(weights, len(weights))

    # Write the data to text 
    np.savetxt('weights.txt', weights)

    #write the number of labels and the dictionary in a .txt file
    with open('variables_to_pass.txt', 'w') as convert_file: 
        convert_file.write(str(len(weights)) + '\n')
        convert_file.write(json.dumps(label2idx))

    test['gene_id'] = test.index
    train.to_parquet(data_path + 'train_dataset.parquet', index=False)
    validation.to_parquet(data_path + 'validation_dataset.parquet', index=False)
    test.to_parquet(data_path + 'test_dataset.parquet', index=False)