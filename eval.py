import torch
from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
from utils_transform import *
from transformer import *
from torchnlp.metrics import get_moses_multi_bleu
import spacy
import pdb
import argparse

# Set up parser for arguments
parser = argparse.ArgumentParser(description='Evaluating performance of a model')
parser.add_argument('--vocab', type=int, default=10000, help='Vocab size. MUST MATCH')
parser.add_argument('--batch', type=int, default=512, help='Batch size')
parser.add_argument('--path', type=str, default="save", help='model path within models/ directory')
parser.add_argument('--eval', type=str, default="accuracy", help='type of eval to do: accuracy, bleu, all')
parser.set_defaults(context=False)
args = parser.parse_args()
print("Command line arguments: {%s}" % args)

VOCAB_SIZE = args.vocab
BATCH_SIZE = args.batch
train_path = "train_200k.csv" # TRAIN PATH MUST MATCH ORIGINAL
SOS, EOS, PAD, BOS = "<s>", "</s>", "<pad>", "<bos>" # Represents begining of context sentence

if torch.cuda.device_count() > 0:
  print('GPUs available:', torch.cuda.device_count())
  model.cuda()
  criterion.cuda()
  device = torch.device('cuda', 0)
else:
  device = torch.device('cpu')


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
    tr_voc = VOCAB_SIZE + 2
    en_voc = VOCAB_SIZE + 4
    model = make_model(tr_voc, en_voc, N=6)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def greedy_decode(model, batch, max_len, start_symbol):
    src = batch.src, src_mask = batch.src_mask # This is just wrong lol
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    total_prob = 0.0
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           ys.clone().detach(),
                           (subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        max_prob, next_word = torch.max(prob, dim = 1)
        total_prob += max_prob.data
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def beam_decode(model, src, src_mask, src_context, pad_idx, max_len, start_symbol, end_symbol, k=5):
    pdb.set_trace()
    batch = Batch(src.unsqueeze(0), src_mask.unsqueeze(0), src_context.unsqueeze(0), pad_idx)
    memory = model.encode(batch)
    ys = torch.ones(1,1).fill_(start_symbol).type_as(src.data)
    hypotheses = [(ys, 0.0)]
    for i in range(max_len):
      candidates_at_length = []
      for hypothesis, previous_prob in hypotheses:
        if hypothesis[0, -1] == end_symbol:
          candidates_at_length.append((hypothesis, previous_prob))
        else:
          # feed through model
          out = model.decode(memory, src_mask, 
                               hypothesis.clone().detach(),
                               (subsequent_mask(hypothesis.size(1))
                                        .type_as(src.data)))
          probs = model.generator(out[:, -1])
          # Keep track of top k predictions for each candidates
          top_probs, predictions_at_step = torch.topk(probs, k, dim=1)
          new_hypotheses = [torch.cat([hypothesis.clone(), pred.reshape(1,1)], dim=1) for pred in predictions_at_step.flatten()]
          new_probs = top_probs.flatten().data + previous_prob
          candidates_at_length = candidates_at_length + list(zip(new_hypotheses, new_probs))
      hypotheses = sorted(candidates_at_length, key = lambda x: x[1], reverse=True)[:k]
    return hypotheses[0]

lambda str_of_tokens: [EN.vocab.itos[i] for i in tokenized]

def eval_bleu(pad_idx, eval_iter, model, max_len, start_symbol, end_symbol):
  bleus = []
  for b in eval_iter:
    b = rebatch(pad_idx, b)
    for i in range(b.src.size(0)): # batch_size
      hypothesis = beam_decode(model, b.src[i], b.src_mask[i], b.src_context[i],
       pad_idx, max_len, start_symbol, end_symbol, k=5)[1:] # cut off SOS
      targets = b.trg_y # doesn't have SOS
      hyp_str = str_of_tokens(hypothesis).join(" ")
      trg_str = str_of_tokens(targets).join(" ")
      pdb.set_trace()
      bleu = get_moses_multi_bleu([hyp_str], [trg_str])
      bleus.append(bleu)
  return sum(bleus) / len(bleus)

def eval_accuracy(pad_idx, eval_iter, model):
  n_correct = 0.0
  n_total = 0.0
  for b in eval_iter:
    batch_correct, batch_incorrect = rebatch_for_eval(pad_idx, b)
    probs = torch.stack([
      log_likelihood(model, batch_correct), 
      log_likelihood(model, batch_incorrect)], dim=1) # n x 2
    correct = probs[:, 0] > probs[:, 1] # should assign higher probability to the left
    n_correct += torch.sum(correct).item()
    n_total += correct.size(0)
  print("Correct: %d / %d = %f" % (n_correct, n_total, (n_correct / n_total)))
  return

def eval_discriminative(pad_idx, path_to_test_set, model):
  full_path = "data/%s" % (path_to_test_set)
  print('Evaluating discriminative dataset: %s ' % (full_path))
  test = TabularDataset(
    full_path,
    format='tsv', 
    fields=data_fields)

  test_iter = Iterator(
    test, batch_size=100, sort_key=lambda x: 1, repeat=False, train=False)

  eval_discrim(pad_idx, test_iter, model)



print('loading spacy')
en = spacy.load('en')
print('finished')
def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

# Context and source / target fields for English + Turkish
# TODO:CHECK THIS ## NEED TO FLIP ONCE U GET NEW MODEL
TR = Field(init_token = SOS, eos_token = EOS, lower = True, pad_token=PAD) 
EN = Field(tokenize=tokenize_en, lower=True, pad_token=PAD)

# Must be in order
# !!! IMPT note there are five fields
data_fields = [
  ('src_context', TR), ('src', TR),
  ('trg_context', EN), ('trg_correct', EN), ('trg_incorrect', EN)]

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
MIN_FREQ = 5
TR.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
EN.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
pad_idx = EN.vocab.stoi[PAD]

print("TR vocab size: %d, EN vocab size: %d" % (len(TR.vocab), len(EN.vocab)))
print('Done building vocab')

print("Loading model...")
model = load('models/' + args.path)
print("Model loaded.")

if args.eval == "accuracy" or args.eval == "all":
  for path in ["pro_stereotype.tsv", "anti_stereotype.tsv", "male_subject.tsv", "female_subject.tsv"]:
    eval_discriminative(pad_idx, path, model)

if args.eval == "bleu" or args.eval == "all":
  eval_bleu(pad_idx, valid_iter, model, 10, EN.vocab.stoi[SOS],EN.vocab.stoi[EOS])