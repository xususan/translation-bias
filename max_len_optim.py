"Script to figure out what max length I should use."

import pdb
import csv
import sys
import argparse
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils_transform import load_bpe


random.seed(1)

parser = argparse.ArgumentParser(description='Create New Datasets (with special properties)')
parser.add_argument('--inpath', type=str, help='Path to infile (within data/)')
parser.add_argument('--vocab_size', type=int, help='max size of vocab to use')
args = parser.parse_args()

bpemb_tr, bpemb_en = load_bpe(args.vocab_size)

df=  pd.read_csv("data/" + args.inpath, sep='\t')

pdb.set_trace()
len_src = df['tr'].apply(bpemb_tr.encode).apply(len)
len_tgt = df['en'].apply(bpemb_en.encode).apply(len)

del df

lm= LinearRegression()
lm.fit(len_src.reshape(-1, 1), len_tgt.reshape(-1, 1))

print("Coef:, ", lm.coef_)
print("Intercept:, ", lm.coef_)