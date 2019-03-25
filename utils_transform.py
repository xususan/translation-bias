import time
import torch
import numpy as np
from transformer import subsequent_mask
from torchtext import data
from torchtext.data import Field, BucketIterator, TabularDataset, Pipeline
from bpemb import BPEmb
import pdb
from torchtext.vocab import Vectors


SOS, EOS, PAD = "<s>", "</s>", "<pad>"
BOC, BOS = "<boc>", "<bos>"

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

def run_epoch(data_iter, model, loss_compute, multi_gpu):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0.0
    total_loss = 0
    tokens = 0.0
    for i, batch in enumerate(data_iter):
        src, tgt, src_mask, tgt_mask = batch.src, batch.trg, batch.src_mask, batch.trg_mask
        src_context, src_context_mask = batch.src_context, batch.src_context_mask
        if multi_gpu:
            device = torch.device('cuda', 0)
            src.to(device)
            tgt.to(device)
            src_mask.to(device)
            tgt_mask.to(device)
            src_context.to(device)
            src_context_mask.to(device)
        out = model.forward(src, src_mask, src_context, src_context_mask, tgt, tgt_mask)
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
    def __init__(self, generator, criterion, opt=None, multi_gpu=False):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.multi_gpu = multi_gpu
        if self.multi_gpu:
            print("Loss initialized with multi gpu")
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion((x.contiguous().view(-1, x.size(-1))), 
                              (y.float().contiguous().view(-1))) / norm
        if self.multi_gpu:
            loss.sum().backward()
        else:
            loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm

class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[torch.Tensor(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [torch.Tensor(torch.cat(og, dim=1), requires_grad=True) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize

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
            self.train_csv, self.val_csv, self.test_csv = "train_2m.csv", "val_10k.csv", "test_10k.csv"
        else:
            self.vocab_size = 50000
            self.train_csv, self.val_csv, self.test_csv = "train_2m.csv", "val_10k.csv", "test_10k.csv"

        if args.train != "None":
            self.train_csv = args.train

        self.use_bpe = args.bpe
        if self.use_bpe == False:
            print("NOT using BPE.")
        else:
            print("Using BPE. Default setting.")

        self.use_pretrained_embeddings = args.pretrainedembed
        if self.use_pretrained_embeddings:
            print("WARNING: Using pretrained embeddings.")
        else:
            print("Learning embedings as we go. Default settng.")

def load_bpe(vocab_size):
    """ Load pre-trained byte pair embedding models.

    Return src, trg
    """
    bpemb_tr = BPEmb(lang="tr", vs=vocab_size)
    bpemb_en = BPEmb(lang="en", vs=vocab_size)
    return bpemb_tr, bpemb_en

def process_word2vec_file(word2vec_filepath, lang_field):
    vectors = Vectors(name=word2vec_filepath) # model_name + path = path_to_embeddings_file
    return vectors


def load_train_val_test_datasets(params):
    """
    Returns datasets and vocab objects
    """
    # Context and source / target fields for English + Turkish
    # 03/01: need to change lower to True?.
    set_lower=True

    if params.use_bpe:
        bpemb_tr, bpemb_en = load_bpe(params.vocab_size)
        TR_CONTEXT = Field(tokenize=bpemb_tr.encode, 
            lower=set_lower, pad_token=PAD, init_token=BOC)
        TR_SRC = Field(tokenize=bpemb_tr.encode, 
            lower=set_lower, pad_token=PAD, init_token=BOS)
        EN = Field(tokenize=bpemb_en.encode, 
            lower=set_lower, pad_token=PAD, init_token=SOS, eos_token=EOS)
        data_fields = [
          ('src_context', TR_CONTEXT), ('src', TR_SRC),
          ('trg_context', EN), ('trg', EN)]
    else:
        # en = spacy.load('en')
        # def tokenize_en(sentence):
        #     return [tok.text for tok in en.tokenizer(sentence)]

        # SOS, EOS, PAD, BOS = "<s>", "</s>", "<pad>", "<bos>" # Represents begining of context sentence
        # Context and source / target fields for English + Turkish
        TR = Field(lower=True, pad_token=PAD)
        EN = Field(lower=True, pad_token=PAD, init_token = SOS, eos_token =EOS)
        TR_SRC = None; TR_CONTEXT = None
        data_fields = [
        ('src_context', TR), ('src', TR),
        ('trg_context', EN), ('trg', EN)]

    print("lower = %r" % set_lower)

    # Must be in order
    

    train, val, test = TabularDataset.splits(
      path='data/', 
      train=params.train_csv,
      validation=params.val_csv,
      test=params.test_csv,
      format='tsv', 
      fields=data_fields)

    print('Building vocab...')
    MIN_FREQ = 1
    if TR_SRC and TR_CONTEXT:
        print("Sharing embeddings")
        TR_SRC.build_vocab(train.src, train.src_context, min_freq=MIN_FREQ, max_size=params.vocab_size)
        TR_CONTEXT.vocab = TR_SRC.vocab
        TR = TR_SRC
    else:
        print("not splitting tr_src/ context")
        TR.build_vocab(train, min_freq=MIN_FREQ, max_size=params.vocab_size)

    EN.build_vocab(train, min_freq=MIN_FREQ, max_size=params.vocab_size)

    if params.use_pretrained_embeddings:
        assert(not(params.use_bpe))
        # debiased_vectors_path = "data/embeddings/vectors.w2v.debiased.txt"
        # print("Loading debiased vectors from .. %s" % debiased_vectors_path)
        # embeds = load_glove_embeddings(debiased_vectors_path, EN.vocab.stoi)
        # vectors = process_word2vec_file(debiased_vectors, EN)
        # EN.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)



    print("TR=TR_SRC vocab size: %d, EN vocab size: %d" % (len(TR.vocab), len(EN.vocab)))
    print('Done building vocab')


    return train, val, test, TR, EN




