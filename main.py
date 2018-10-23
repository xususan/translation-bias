from torchtext import data, datasets
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import Seq2Seq
import os.path
import pdb
import argparse
import utils

BOS_WORD = '<s>'
EOS_WORD = '</s>'
MAX_LEN = 20
BATCH_SIZE = 32

DE = data.Field(tokenize=utils.tokenize_de)
EN = data.Field(tokenize=utils.tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

# Set up parser for arguments
parser = argparse.ArgumentParser(description='Translation')
parser.add_argument('--attn', type=bool, default=False, help='use attention')
parser.add_argument('--model_path', type=str, default=None, help='load a model')
parser.add_argument('--epochs', type=int, default=5, help='num epochs, default 5')
parser.add_argument('--n_layers', type=int, default=1, help='num layers, default 1')
args = parser.parse_args()

# Download dataset, build vocab
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                         len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 5
DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)

print("Finish build vocab")

train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                  repeat=False, sort_key=lambda x: len(x.src))

print("Done bucketing data")

model = Seq2Seq(hidden_size=50, input_vocab_size=len(DE.vocab), output_vocab_size=len(EN.vocab))
if torch.cuda.is_available(): model = model.cuda()
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=0.5)
criterion = nn.CrossEntropyLoss()


utils.train(train_iter, val_iter, model, nn.CrossEntropyLoss(), optimizer, args.epochs)
