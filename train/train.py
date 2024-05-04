
#Hugging face trainer model
# -> multi-processing GPUs (we should have two of the same machine)


### Things to do in this script ###
#Créer une grande fonction train (comme pour ProtTrans)
#1) compter le nombre de paramètres du modèle
#utiliser le même tokenizer
#charger les poids du modèle pré-etnrai

import os 

import os.path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import re
import numpy as np
import pandas as pd
import polars as pl


from evaluate import load
from datasets import Dataset

from tqdm import tqdm
import random

from scipy import stats
from sklearn.metrics import accuracy_score

#from settings import max_length, loss_type, weighted, batch_size

# Set environment variables to run Deepspeed from a notebook
#os.environ["MASTER_ADDR"] = "localhost"
#os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
#os.environ["RANK"] = "0"
#os.environ["LOCAL_RANK"] = "0"
#os.environ["WORLD_SIZE"] = "1"
#os.environ['DS_SKIP_CUDA_CHECK'] = '1'
#os.environ["WANDB_PROJECT"]="Fine-tuning_ProtTrans_LoRA_V1"


print("Torch version: ",torch.__version__)
print("Cuda version: ",torch.version.cuda)
print("Numpy version: ",np.__version__)
print("Pandas version: ",pd.__version__)
print("Transformers version: ",transformers.__version__)
print("Datasets version: ",datasets.__version__)

"""
Torch version:  1.13.1
Cuda version:  11.7
Numpy version:  1.22.3
Pandas version:  1.5.3
Transformers version:  4.26.1
Datasets version:  2.9.0
"""


# Set random seeds for reproducibility of your trainings run
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

# Dataset creation
def create_dataset(tokenizer,seqs,labels, max_length):
    tokenized = tokenizer(seqs, max_length=max_length, padding=True, truncation=True) #changed padding to True by default
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", labels)

    return dataset



#Saving only fine-tuned weights
class Save_LoRA_Callback(TrainerCallback):
    """
    Allows to save only finetuned weights inside the Trainer function during training. (10 mbs instead of 2 gigas for T5)
    """
    def __init__(self, path):
        self.path = path

    def on_epoch_end(self, args, state, control, model, **kwargs): #each time training part in epoch ends, this function is executed
        if args.local_rank == 0:
            file_list = os.listdir(os.path.dirname(self.path))
            epochs = [int(re.search(r'\d+', filename).group()) for filename in file_list if re.search(r'\d+', filename)] #looking for last epoch
        
            if epochs:
                last_epoch = max(epochs)
            else: 
                last_epoch = 0

            #writing weights only for non frozen parameters
            save_model(model, filepath=self.path.replace('.',f'_epoch_{last_epoch+1}.'))  
        
            print(f"Weights of {last_epoch+1}th epoch saved")


# Main training fuction
def train_per_protein(
        train_df,         #training data
        valid_df,         #validation data      
        num_labels= 1,    #1 for regression, >1 for classification
        loss_type = 'cross-entropy',
        loss_weights=None,
    
        # effective training batch size is batch * accum
        # we recommend an effective batch size of 8 
        batch= 4,         #for training
        accum= 2,         #gradient accumulation

        val_batch = 16,   #batch size for evaluation
        epochs= 10,       #training epochs
        lr= 3e-4,         #recommended learning rate
        seed= 42,         #random seed
        gpu= 1,          #gpu selection (1 for first gpu)
        max_length = 1024,
    
        save_path = None,
        load_last_finetuned_weights= True):        

    # Set gpu device
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu-1)
    
    # Set all random seeds
    set_seeds(seed)
    
    print("Model to be loaded")
    # load model
    if load_last_finetuned_weights:
        path_weights = os.path.dirname(save_path)
        filtered_paths = [path for path in os.listdir(path_weights) if path.endswith('.pth')]
        
        if filtered_paths:
            latest_epoch_weights = max(filtered_paths, key=lambda path: int(path.split('_')[-1].split('.')[0]), default=None)
            print(f"Resuming training with the {latest_epoch_weights} weights.")
            model, tokenizer = load_model(f"{path_weights}/{latest_epoch_weights}", num_labels=num_labels, mixed=True, loss_type=loss_type, loss_weights=loss_weights)
        else:
            model, tokenizer = PT5_classification_model(num_labels=num_labels, half_precision=True, loss_type=loss_type, loss_weights=loss_weights)
    else:
        model, tokenizer = PT5_classification_model(num_labels=num_labels, half_precision=True, loss_type=loss_type, loss_weights= loss_weights)

    print("Model loaded")

    # Preprocess inputs
    # Replace uncommon AAs with "X"
    train_df["AA_seq"]=train_df["AA_seq"].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True) 
    valid_df["AA_seq"]=valid_df["AA_seq"].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True) 
    # Add spaces between each amino acid for PT5 to correctly use them
    train_df['AA_seq']=train_df.apply(lambda row : " ".join(row["AA_seq"]), axis = 1)
    valid_df['AA_seq']=valid_df.apply(lambda row : " ".join(row["AA_seq"]), axis = 1)

    # Create Datasets
    train_set=create_dataset(tokenizer,list(train_df['AA_seq']),list(train_df['label']), max_length = max_length)
    valid_set=create_dataset(tokenizer,list(valid_df['AA_seq']),list(valid_df['label']), max_length = max_length)

    # Huggingface Trainer arguments
    args = TrainingArguments(
        "./",
        evaluation_strategy = "epoch",
        logging_strategy = "epoch",
        save_strategy = "no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed = seed,
        fp16 = mixed,
        report_to="wandb",

        
    ) 

    # Metric definition for validation data
    def compute_metrics(eval_pred):
        if num_labels>1:  # for classification
            metric = load("accuracy")
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
        else:  # for regression
            metric = load("spearmanr")
            predictions, labels = eval_pred

        return metric.compute(predictions=predictions, references=labels)
    

    # Trainer          
    trainer = Trainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[Save_LoRA_Callback(save_path)])    
    
    print("On va commencer l'entrainement")
    # Train model
    trainer.train()

    return tokenizer, model, trainer.state.log_history


#fetching number of labels from preprocessing to have the correct number of heads
with open("variables_to_pass.txt", "r") as file:
    num_labels = file.readline().strip('\n')
num_labels = int(num_labels)
print(f"Working with {num_labels} classes using {loss_type} loss, weighted : {weighted}")

#Relative paths using variables in the run.sh file
data_path = sys.argv[2] + '/data/'
os.system(f"mkdir -p {sys.argv[2]}/ProtTrans/LoRA_Checkpoints_{loss_type}")
save_path = sys.argv[2] + f'/ProtTrans/LoRA_Checkpoints_{loss_type}/LoRA_finetuned.pth'

print(sys.argv[2], sys.argv[3])

num_gpus = sys.argv[3]
print(num_gpus)
num_gpus = int(num_gpus)

if weighted:
    weights = np.loadtxt('weights.txt')
    weights = torch.FloatTensor(weights)
else:
    weights = None

print(f"Training on {num_gpus} GPUs")

print("Reading the data")
train_dataset, validation_dataset = pd.read_parquet(data_path + 'train_dataset.parquet'), pd.read_parquet(data_path + 'validation_dataset.parquet')

print("Launching training")
print(f"Protein maximum length is {max_length}")

tokenizer, model, history = train_per_protein(train_dataset, validation_dataset, num_labels=num_labels, loss_type=loss_type, loss_weights=weights, batch=batch_size, accum=8, epochs=5, seed=42, bool_deepspeed=True, gpu=num_gpus, max_length=max_length, save_path=save_path, load_last_finetuned_weights=True)

print("Model trained")

