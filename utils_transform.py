import time
import torch
import numpy as np
from transformer import subsequent_mask
from torchtext import data
from torchtext.data import Field, BucketIterator, TabularDataset, Pipeline
from bpemb import BPEmb
import pdb

SOS, EOS, PAD, BOS = "<s>", "</s>", "<pad>", "<bos>" 

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, src_context=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
        self.src_context = src_context
        self.src_context_mask = (src_context != pad).unsqueeze(-2)
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & (
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0.0
    total_loss = 0
    tokens = 0.0
    for i, batch in enumerate(data_iter):
        pdb.set_trace()
        out = model.forward(batch)
        batch_ntokens = batch.ntokens.float()
        loss = loss_compute(out, batch.trg_y, batch_ntokens)
        total_loss += loss
        total_tokens += batch_ntokens
        tokens += batch_ntokens
        if i % 1000 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch_ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0.0
    return total_loss / total_tokens

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion((x.contiguous().view(-1, x.size(-1))), 
                              (y.float().contiguous().view(-1))) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    # Originally, src is [src_len x batch_size]
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    src_context = batch.src_context.transpose(0,1)
    return Batch(src, trg, src_context, pad_idx)

pad_date = lambda num: '0' * (2 - len(str(num))) + str(num)

class Params:
    "Object for holding training parameters."
    def __init__(self, args):
        if args.size == "mini":
            self.vocab_size = 1000
            self.train_csv, self.val_csv, self.test_csv = "train_mini.csv", "val_mini.csv", "test_mini.csv"
        elif args.size == "mid":
            self.vocab_size = 10000
            self.train_csv, self.val_csv, self.test_csv = "train_200k.csv", "val_10k.csv", "test_10k.csv"
        else:
            self.vocab_size = 50000
            self.train_csv, self.val_csv, self.test_csv = "train_2m.csv", "val_10k.csv", "test_10k.csv"

def load_bpe(vocab_size):
    """ Load pre-trained byte pair embedding models.

    Return src, trg
    """
    bpemb_tr = BPEmb(lang="tr", vs=vocab_size)
    bpemb_en = BPEmb(lang="en", vs=vocab_size)
    return bpemb_tr, bpemb_en

def load_train_val_test_datasets(args, params):
    """
    Returns datasets and vocab objects
    """
    # Context and source / target fields for English + Turkish
    bpemb_tr, bpemb_en = load_bpe(params.vocab_size)
    TR = Field(Pipeline(bpemb_tr.encode), 
        lower=False, pad_token=PAD)
    EN = Field(Pipeline(bpemb_en.encode), 
        lower=False, pad_token=PAD, init_token=SOS, eos_token=EOS)

    # Must be in order
    data_fields = [
      ('src_context', TR), ('src', TR),
      ('trg_context', EN), ('trg', EN)]

    train, val, test = TabularDataset.splits(
      path='data/', 
      train=params.train_csv,
      validation=params.val_csv,
      test=params.test_csv,
      format='tsv', 
      fields=data_fields)

    print('Building vocab...')
    MIN_FREQ = 1
    TR.build_vocab(train, min_freq=MIN_FREQ, max_size=params.vocab_size)
    EN.build_vocab(train, min_freq=MIN_FREQ, max_size=params.vocab_size)
    print("TR vocab size: %d, EN vocab size: %d" % (len(TR.vocab), len(EN.vocab)))
    print('Done building vocab')

    return train, val, test, TR, EN





