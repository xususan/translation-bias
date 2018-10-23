import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pdb

USE_CUDA = torch.cuda.is_available()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding_size = hidden_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.dropout_p = 0
        self.n_layers = 1
        self.rnn = nn.LSTM(
            self.embedding_size, 
            hidden_size, 
            num_layers=self.n_layers, 
            dropout=self.dropout_p)

    def forward(self, input):
        batch_size = input.size(1)
        embedded = self.embedding(input)
        h_0 = self.init_hidden(batch_size)
		pdb.set_trace()
		output, hidden = self.rnn(embedded, h_0)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden =  torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if USE_CUDA: hidden = hidden.cuda()
        return (hidden, hidden.clone())

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output) # Try other?
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        # output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size,hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_vocab_size)


    def forward(self, source, target):
        if USE_CUDA: source = source.cuda()

        # Encode
        output_encoder, hidden_encoder = self.encoder(source)

        # Decode
        output_decoder, hidden_decoder = self.decoder(target, hidden_encoder)

        # Predict
        return output_decoder
