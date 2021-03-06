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
parser.add_argument('--train', type=str, default="train_2m.csv", help='Train path. MUST MATCH THE FILE USED IN TRAINING')
parser.add_argument('--val', type=str, default="val_10k.csv", help='BLEU validation test path, if different from val_10k.csv')
parser.add_argument('--batch', type=int, default=512, help='Batch size')
parser.add_argument('--path', type=str, default="save", help='model path within models/ directory')
parser.add_argument('--eval', type=str, default="accuracy", help='type of eval to do: accuracy, bleu, all')
parser.add_argument('--context', dest='context', action='store_true')
parser.add_argument('--no-context', dest='context', action='store_false')
parser.add_argument('--bpe', dest='bpe', action='store_true')
parser.add_argument('--no-bpe', dest='bpe', action='store_false')
parser.add_argument('--pretrained-embed', dest='pretrainedembed', action='store_true')
parser.add_argument('--no-pretrained-embed', dest='pretrainedembed', action='store_false')
parser.set_defaults(context=False)
parser.set_defaults(bpe=True)
parser.set_defaults(pretrainedembed=False)
args = parser.parse_args()
print("Command line arguments: {%s}" % args)

VOCAB_SIZE = args.vocab
BATCH_SIZE = args.batch
if args.vocab == 10000:
  train_path, val_path, test_path= "train_2m.csv", "val_10k.csv", "test_10k.csv" # TRAIN PATH MUST MATCH ORIGINAL
  train_path = args.train
  print("Set training path to %s" % args.train)
  val_path = args.val
  print("Set validation path (for BLEU) to %s" % args.val)
elif args.vocab == 1000:
  train_path, val_path, test_path = "train_mini.csv", "val_mini.csv", "test_mini.csv"
else:
  print("Args vocab wasn't 1000 or 10000.")
SOS, EOS, PAD = "<s>", "</s>", "<pad>" # Represents begining of context sentence
BOC, BOS = "<boc>", "<bos>"

device = torch.device('cpu')

if args.bpe:
  bpemb_tr, bpemb_en = load_bpe(VOCAB_SIZE)
else:
  bpemb_en = None; bpemb_tr = None
# Context and source / target fields for English + Turkish
# Lower = true

## SUPER IMPORTANT THAT THE VOCAB FIELDS MATCH

date_model_trained = int(args.path[:4])

if date_model_trained < 301:
  print("Model trained before March 1. USING OLD VERSION OF VOCAB")
  USE_NEW_DOUBLE_TR = False
else:
  print("Model trained after march 1. USING NEW VERSION OF VOCAB")
  USE_NEW_DOUBLE_TR = True



if USE_NEW_DOUBLE_TR and (args.bpe):
  print("Using new version of vocab. BPE is true.")
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
elif args.bpe:
  print("Using old version of vocab WITH BPE")
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
    validation=val_path,
    test=test_path,
    format='tsv', 
    fields=[('src_context', TR), ('src', TR),
    ('trg_context', EN), ('trg', EN)])
  print('done')
else:
  print("Using non-BPE vocab.")
  TR = Field(lower=True, pad_token=PAD)
  EN = Field(lower=True, pad_token=PAD, init_token = SOS, eos_token =EOS)
  data_fields = [
  ('src_context', TR), ('src', TR),
  ('trg_context', EN), ('trg', EN)]
  train, val, test = TabularDataset.splits(
      path='data/', 
      train=train_path,
      validation=val_path,
      test=test_path,
      format='tsv', 
      fields=data_fields)
  #TR.build_vocab(train, min_freq=MIN_FREQ, max_size=params.vocab_size)
  #EN.build_vocab(train, min_freq=MIN_FREQ, max_size=params.vocab_size)


print("Building vocab...")

MIN_FREQ = 1
if USE_NEW_DOUBLE_TR and args.bpe:
  TR_CONTEXT.build_vocab(train.src, train.src_context, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
  TR_SRC.vocab = TR_CONTEXT.vocab
  TR = TR_SRC
else:
  TR.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)

EN.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
pad_idx = EN.vocab.stoi[PAD]

print('making validation iterator')
valid_iter = Iterator(val, batch_size=BATCH_SIZE, device=device,
                      repeat=False, sort=False, train=False) 
print('done')

print("TR vocab size: %d, EN vocab size: %d" % (len(TR.vocab), len(EN.vocab)))
print('Done building vocab')
rev_tokenize_en = lambda tokenized: [EN.vocab.itos[i] for i in tokenized]
rev_tokenize_tr = lambda tokenized: [TR.vocab.itos[i] for i in tokenized]

print("Loading model...")
model = load('models/' + args.path, len(TR.vocab), len(EN.vocab), use_context=args.context, share_embeddings=args.bpe, pretrained_embeddings=False)
print("Model loaded from %s" % args.path)

if args.pretrainedembed:
  assert(not(args.bpe))
  debiased_vectors_path = "data/embeddings/vectors.w2v.debiased.txt"
  print("Loading EN debiased vectors from .. %s" % debiased_vectors_path)
  embeds_en = load_glove_embeddings(debiased_vectors_path, EN.vocab.stoi)
  model.tgt_embed[0].lut.weight = nn.Parameter(embeds_en)
  model.tgt_embed[0].lut.weight.requires_grad = False

  debiased_vectors_path_tr = "data/embeddings/vectors_tr.w2v.debiased.txt"
  print("Loading TR debiased vectors from .. %s" % debiased_vectors_path_tr)
  embeds_tr = load_glove_embeddings(debiased_vectors_path_tr, TR.vocab.stoi)
  model.src_embed[0].lut.weight = nn.Parameter(embeds_tr)
  model.src_embed[0].lut.weight.requires_grad = False

if args.eval == "accuracy" or args.eval == "all":
  for path in ["pro_stereotype.tsv", "anti_stereotype.tsv", "male_subject.tsv", "female_subject.tsv"]:
    eval_accuracy(pad_idx, path, model, TR, EN)

if args.eval == "bleu" or args.eval == "all":
  print("Evaluating BLEU")
  out_path = args.path[:-3] + "_" + val_path
  eval_bleu(pad_idx, valid_iter, model, 30, EN.vocab.stoi[SOS],EN.vocab.stoi[EOS], rev_tokenize_en, bpemb_en, out_path)
