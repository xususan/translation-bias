import torch
from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
from utils_transform import *
from transformer import *
from eval_lib import *
import spacy
import pdb
import argparse
from eval_lib import *

"""
Use this script to evaluate the results of a trained model (checkpoint). 

The architecture of the model (transformer.py) MUST be the same as training.

ONLY COMPATIBLE WITH MODELS TRAINED AFTER 2/11.

Notes on vocab: the SRC language MUST have two additional tokens (unk / pad), 
TRG language MUST have four additional tokens (EOS / BOS / unk / pad)

Also, you must build the vocab in the exact same way it was built during training
to ensure that you have the same tokenizer. At least for now, which is kinda sus.

CPU ONLY.
"""
# Set up parser for arguments
parser = argparse.ArgumentParser(description='Evaluating performance of a model')
parser.add_argument('--vocab', type=int, default=10000, help='vocab size, must match max size in build_vocab')
parser.add_argument('--batch', type=int, default=512, help='Batch size')
parser.add_argument('--path', type=str, default="save", help='model path within models/ directory')
parser.add_argument('--eval', type=str, default="accuracy", help='type of eval to do: accuracy, bleu, all')
parser.add_argument('--context', dest='context', action='store_true')
parser.add_argument('--no-context', dest='context', action='store_false')
parser.set_defaults(context=False)
args = parser.parse_args()
print("Command line arguments: {%s}" % args)

VOCAB_SIZE = args.vocab
BATCH_SIZE = args.batch
if args.vocab == 10000:
  train_path, val_path, test_path= "train_200k.csv", "val_10k.csv", "test_10k.csv" # TRAIN PATH MUST MATCH ORIGINAL
elif args.vocab == 1000:
  train_path, val_path, test_path = "train_mini.csv", "val_mini.csv", "test_mini.csv"
else:
  print("Args vocab wasn't 1000 or 10000.")
SOS, EOS, PAD = "<s>", "</s>", "<pad>" # Represents begining of context sentence
BOC, BOS = "<boc>", "<bos>"

device = torch.device('cpu')

bpemb_tr, bpemb_en = load_bpe(VOCAB_SIZE)

# Context and source / target fields for English + Turkish
# Lower = true
TR_CONTEXT = Field(tokenize=bpemb_tr.encode, 
        lower=True, pad_token=PAD, init_token=BOC)
TR_SRC = Field(tokenize=bpemb_tr.encode, 
    lower=True, pad_token=PAD)
EN = Field(tokenize=bpemb_en.encode, 
    lower=True, pad_token=PAD, init_token=SOS, eos_token=EOS)

print('reading in tabular dataset')
train, val, test = TabularDataset.splits(
  path='data/', 
  train=train_path,
  validation=val_path,
  test=test_path,
  format='tsv', 
  fields=[('src_context', TR_CONTEXT), ('src', TR_SRC),
  ('trg_context', EN), ('trg', EN)])
print('finished')

print('making iterator')
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg), len(x.src_context)),
                        batch_size_fn=batch_size_fn, train=False)
print('done')

print("Building vocab...")
MIN_FREQ = 1
TR_CONTEXT.build_vocab(train.src, train.src_context, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
TR_SRC.vocab = TR_CONTEXT.vocab
TR = TR_CONTEXT
EN.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
pad_idx = EN.vocab.stoi[PAD]

print("TR vocab size: %d, EN vocab size: %d" % (len(TR.vocab), len(EN.vocab)))
print('Done building vocab')
rev_tokenize_en = lambda tokenized: [EN.vocab.itos[i] for i in tokenized]
rev_tokenize_tr = lambda tokenized: [TR_SRC.vocab.itos[i] for i in tokenized]

print("Loading model...")
model = load('models/' + args.path, len(TR.vocab), len(EN.vocab), args.context)
print("Model loaded from %s" % args.path)

if args.eval == "accuracy" or args.eval == "all":
  for path in ["pro_stereotype.tsv", "anti_stereotype.tsv", "male_subject.tsv", "female_subject.tsv"]:
    eval_accuracy(pad_idx, path, model, TR, EN)

if args.eval == "bleu" or args.eval == "all":
  print("Evaluating BLEU")
  out_path = args.path[:-3] + "_" + val_path
  eval_bleu(pad_idx, valid_iter, model, 30, EN.vocab.stoi[SOS],EN.vocab.stoi[EOS], rev_tokenize_en, bpemb_en, out_path)
