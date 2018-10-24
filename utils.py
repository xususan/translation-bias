import spacy
import torch
import pdb
from numpy import exp

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


class AverageLosses:
    def __init__(self):
        self.sum_losses, self.count, self.avg = 0.0, 0.0, None

    def update(self, loss, n_obs=1):
        self.sum_losses += loss * n_obs
        self.count += n_obs
        self.avg = self.sum_losses / self.count

def train_batch(model, batch, criterion, optimizer):
    model.zero_grad()
    
    src, trg = batch.src, batch.trg
    if torch.cuda.is_available():
        src = src.cuda()
        trg = trg.cuda()

    scores = model(src, trg)

    # Scores are [len x batch_size x output_vocab_size]
    # Targets are [len x batch_size]

    # Remove <s> from beginning of target
    targets = trg[1:]
    # Remove </s> from end of source bc nothing to predict after that.
    scores = scores[:-1]

    # Reshape.
    new_scr = scores.view(scores.size(0) * scores.size(1), -1)
    new_trg = targets.view(new_scr.size(0))

    loss = criterion(new_scr, new_trg)
    loss.backward()
    optimizer.step()
    return loss.data[0]
    
def validate(model, val_iter, criterion):
    ''' Calculate perplexity on validation set.'''
    model.eval()

    AL = AverageLosses()

    for i, batch in enumerate(val_iter):
        src, trg = batch.src, batch.trg
        if torch.cuda.is_available():
            src = src.cuda()
            trg = trg.cuda()

        scores = model(src, trg)
        # Remove <s> from beginning of target
        targets = trg[1:]
        # Remove </s> from end of source bc nothing to predict after that.
        scores = scores[:-1]

        # Reshape.
        new_scr = scores.view(scores.size(0) * scores.size(1), -1)
        new_trg = targets.view(new_scr.size(0))

        loss = criterion(new_scr, new_trg)

        # Count number of non-padding elements on target.
        num_words = (new_trg != 1).sum().data[0]

        AL.update(loss.data[0], n_obs=num_words)

    return exp(AL.avg)

def train(train_iter, val_iter, model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        AL = AverageLosses()
        for i, batch in enumerate(train_iter):
            loss = train_batch(model, batch, criterion, optimizer)
            AL.update(loss)
            
            if i % 1000 == 10:
                print('''Epoch [{e}/{num_e}]\t Batch [{b}/{num_b}]\t Loss: {l:.3f}'''.format(e=epoch+1, num_e=num_epochs, b=i, num_b=len(train_iter), l=AL.avg))

        ppl = validate(model, val_iter, criterion)
        print('''Epoch [{e}/{num_e}]\t Perplexity: {ppl:.3f}'''.format(e=epoch+1, num_e=num_epochs, ppl=ppl))

