from genericpath import exists
import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from tqdm import tqdm

import file_handling as fh

parser = argparse.ArgumentParser()
parser.add_argument("input_vocab", type=str, help="input vocab file")
parser.add_argument(
        "--teacher-vocab",
        dest="teacher_vocab",
        type=str,
        help="Teacher vocab list",
    )
parser.add_argument(
        "--model-file",
        dest="model_file",
        type=str,
        help="Pretrained model file",
    )
parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        help="Output directory",
    )
parser.add_argument(
        "--emb-dim",
        type=int,
        default=300,
        help="Dimension of input embeddings",
    )

options = parser.parse_args()

input_vocab_file = options.input_vocab
teacher_vocab_file = options.teacher_vocab
model_file = options.model_file
output_dir = options.output_dir
emb_dim = options.emb_dim 

# load the vocab list
input_vocab = fh.read_json(input_vocab_file)
teacher_vocab = fh.read_json(teacher_vocab_file)
input_teacher_and = set(input_vocab) & set(teacher_vocab)
input_vocab_size = len(input_vocab)
teacher_vocab_size = len(teacher_vocab)
and_vocab_size = len(input_teacher_and)
print("input:", input_vocab_size)
print("teacher:", teacher_vocab_size)
print("lost words:", input_vocab_size - and_vocab_size )

# load teacher embeddings
model_params = torch.load(model_file)
teacher_emb = model_params["model_state_dict"]['embeddings_x.background']
teacher_emb = teacher_emb.to('cpu').detach().numpy().copy() 

# init input embeddings
input_emb = torch.zeros((emb_dim, input_vocab_size))
kaiming_uniform_(input_emb, a=np.sqrt(5))
xavier_uniform_(input_emb)
input_emb = input_emb.to('cpu').detach().numpy()

input_df = pd.DataFrame(input_emb, columns=input_vocab)
teacher_df = pd.DataFrame(teacher_emb, columns=teacher_vocab)
print("input\n", input_df.head())
print("teacher\n", teacher_df.head())

# Update input embeddings for each word(column)
print("Updating embeddings")
for word in tqdm(input_vocab):
    if word in teacher_vocab:
        input_df[word] = teacher_df[word]

print("updated emb\n", input_df.head())

# save pretreined embeddings to npy file
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "pretrained_emb"), input_df.values)
