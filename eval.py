import torch
from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
from utils_transform import *
from transformer import *
from torchnlp.metrics import get_moses_multi_bleu
import spacy
import pdb

VOCAB_SIZE = 20
train_path = "train_mini.csv" # TRAIN PATH MUST MATCH ORIGINAL

hypotheses = ["The brown fox jumps over the dog 笑"]
references = ["The quick brown fox jumps over the lazy dog 笑"]

# Compute BLEU score with the official BLEU perl script
# print(get_moses_multi_bleu(hypotheses, references, lowercase=True))  # RETURNS: 47.9

def rebatch_for_eval(pad_idx, batch):
  """Returns two batches: one where batch.trg, batch.trg_y match the correct
  translation; and one where they match the incorrect translation.
  """
  src = batch.src.transpose(0, 1)
  src_context = batch.src_context.transpose(0,1)
  trg_correct = batch.trg_correct.transpose(0,1)
  trg_incorrect = batch.trg_incorrect.transpose(0,1)
  return Batch(src, trg_correct, src_context, pad_idx), Batch(src, trg_incorrect, src_context, pad_idx)

def log_likelihood(model, batch):
    memory = model.encode(batch)
    total_prob = torch.zeros(batch.trg_y.size(0))
    for i in range(0, batch.trg_y.size(1)): # trg_len
        y_prev = batch.trg[:, :i + 1]
        out = model.decode(memory, batch.src_mask, 
                           y_prev.clone().detach(),
                           (subsequent_mask(y_prev.size(1))
                                    .type_as(batch.src.data)))
        probs = model.generator(out[:, -1]) # batch x vocab
        trg_index = batch.trg_y[:, i]
        prob_of_trg = probs.gather(1, trg_index.view(-1,1)) # not sure about this
        total_prob += prob_of_trg.squeeze()
    return total_prob

def load(path):
    tr_voc = VOCAB_SIZE + 4
    en_voc = VOCAB_SIZE + 2
    model = make_model(tr_voc, en_voc, N=6)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def eval(pad_idx, eval_iter, model):
  n_correct = 0.0
  n_total = 0.0
  for b in eval_iter:
    batch_correct, batch_incorrect = rebatch_for_eval(pad_idx, b)
    probs = torch.stack([
      log_likelihood(model, batch_correct), 
      log_likelihood(model, batch_incorrect)], dim=1) # n x 2
    pdb.set_trace()
    correct = probs[:, 0] > probs[:, 1] # should assign higher probability to the left
    n_correct += torch.sum(correct)
    n_total += correct.size(0)
  print("Correct: %d / %d = %f" % (n_correct, n_total, n_correct / n_total))
  return


en = spacy.load('en')
def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

SOS, EOS, PAD, BOS = "<s>", "</s>", "<pad>", "<bos>" # Represents begining of context sentence
# Context and source / target fields for English + Turkish
TR = Field(init_token = SOS, eos_token = EOS, lower = True, pad_token=PAD)
EN = Field(tokenize=tokenize_en, lower=True, pad_token=PAD)

# Must be in order
# !!! IMPT note there are five fields
data_fields = [
  ('src_context', TR), ('src', TR),
  ('trg_context', EN), ('trg_correct', EN), ('trg_incorrect', EN)]

train = TabularDataset('data/' + train_path, format='tsv',fields=
  [('src_context', TR), ('src', TR),
  ('trg_context', EN), ('trg', EN)])

pro_stereotype = TabularDataset(
  "data/pro_stereotype.tsv",
  format='tsv', 
  fields=data_fields)

MIN_FREQ = 5
TR.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
EN.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
pad_idx = EN.vocab.stoi[PAD]

pro_stereotype_iter = Iterator(pro_stereotype, batch_size=100, sort_key=lambda x: 1, repeat=False, train=False)
model = load("models/mini_1.pt")
eval(pad_idx, pro_stereotype_iter, model)