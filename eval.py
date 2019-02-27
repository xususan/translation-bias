import torch
from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
from utils_transform import *
from transformer import *
from eval_lib import *
from torchnlp.metrics import get_moses_multi_bleu
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
train_path = "train_200k.csv" # TRAIN PATH MUST MATCH ORIGINAL
SOS, EOS, PAD, BOS = "<s>", "</s>", "<pad>", "<bos>" # Represents begining of context sentence

device = torch.device('cpu')

bpemb_tr, bpemb_en = load_bpe(VOCAB_SIZE)

# Context and source / target fields for English + Turkish
TR = Field(tokenize=bpemb_tr.encode, 
        lower=False, pad_token=PAD)
EN = Field(tokenize=bpemb_en.encode, 
    lower=False, pad_token=PAD, init_token=SOS, eos_token=EOS)
rev_tokenize_en = lambda tokenized: [EN.vocab.itos[i] for i in tokenized]
rev_tokenize_tr = lambda tokenized: [TR.vocab.itos[i] for i in tokenized]

print('reading in tabular dataset')
train, val, test = TabularDataset.splits(
  path='data/', 
  train=train_path,
  validation="val_10k.csv",
  test="test_10k.csv",
  format='tsv', 
  fields=[('src_context', TR), ('src', TR),
  ('trg_context', EN), ('trg', EN)])
print('finished')

print('making iterator')
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg), len(x.src_context)),
                        batch_size_fn=batch_size_fn, train=False)
print('done')

print("Building vocab...")
MIN_FREQ = 1
TR.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
EN.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
pad_idx = EN.vocab.stoi[PAD]

print("TR vocab size: %d, EN vocab size: %d" % (len(TR.vocab), len(EN.vocab)))
print('Done building vocab')

print("Loading model...")
model = load('models/' + args.path, len(TR.vocab), len(EN.vocab), args.context)
print("Model loaded.")

if args.eval == "accuracy" or args.eval == "all":
  for path in ["pro_stereotype.tsv", "anti_stereotype.tsv", "male_subject.tsv", "female_subject.tsv"]:
    eval_accuracy(pad_idx, path, model, TR, EN)

if args.eval == "bleu" or args.eval == "all":
  eval_bleu(pad_idx, valid_iter, model, 30, EN.vocab.stoi[SOS],EN.vocab.stoi[EOS], rev_tokenize_en, bpemb_en)
