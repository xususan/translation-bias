import torch
from torchtext.data import Field, BucketIterator, TabularDataset
from utils_transform import *
from transformer import *
from torchnlp.metrics import get_moses_multi_bleu
import spacy

hypotheses = ["The brown fox jumps over the dog 笑"]
references = ["The quick brown fox jumps over the lazy dog 笑"]

# Compute BLEU score with the official BLEU perl script
assert(get_moses_multi_bleu(hypotheses, references, lowercase=True) == 47.88)  # RETURNS: 47.9

def log_likelihood(model, batch):
    memory = model.encode(batch)
    total_prob = 0.0
    pdb.set_trace()
    for i in range(0, batch.trg_y.size(1)): # trg_len
        pdb.set_trace()
        y_prev = batch.trg[:, :i + 1]
        print(y_prev, y_prev.size())
        out = model.decode(memory, batch.src_mask, 
                           y_prev.clone().detach(),
                           (subsequent_mask(y_prev.size(1))
                                    .type_as(batch.src.data)))
        probs = model.generator(out[:, -1])
        trg_index = batch.trg_y[:, i].item()
        prob_of_trg = probs[:, trg_index].item()
        total_prob += prob_of_trg
    return total_prob

def load(path):
    tr_voc = 24
    en_voc = 22
    model = make_model(tr_voc, en_voc, N=6)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

en = spacy.load('en')

SOS, EOS, PAD, BOS = "<s>", "</s>", "<pad>", "<bos>" # Represents begining of context sentence
# Context and source / target fields for English + Turkish
TR = Field(init_token = SOS, eos_token =EOS, lower=True, pad_token=PAD)
EN = Field(tokenize=tokenize_en, lower=True, pad_token=PAD)

# Must be in order
data_fields = [
  ('src_context', TR), ('src', TR),
  ('trg_context', EN), ('trg', EN)]

model = load("models/mini_1.pt")
fake_batch = Batch(src=torch.LongTensor([10,9,8,7,6]).view(1, -1), 
    trg=torch.LongTensor([1,2,3,4,5]).view(1, -1), 
    src_context=torch.LongTensor([5,5,5]).view(1, -1), pad=0)

pro_stereotype = TabularDataset.splits(
  path='data/', 
  train="pro_stereotype.tsv",
  validation=None,
  test=None,
  format='tsv', 
  fields=data_fields)


print(log_likelihood(model, fake_batch))