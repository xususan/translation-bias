import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pdb

USE_CUDA = torch.cuda.is_available()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout_p):
        super(EncoderRNN, self).__init__()
        self.embedding_size = hidden_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.n_layers = n_layers
        self.rnn = nn.LSTM(
            self.embedding_size, 
            hidden_size, 
            num_layers=self.n_layers, 
            dropout=self.dropout_p)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input):
        batch_size = input.size(1)
        embedded = self.dropout(self.embedding(input))
        h_0 = self.init_hidden(batch_size)
        output, hidden = self.rnn(embedded, h_0)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return (hidden, hidden.clone())

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, dropout_p):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=self.n_layers,
            dropout=self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input))
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        return output, hidden

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, decoder_output, decoder_hidden, encoder_output, encoder_hidden):
        # Query is decoder state
        # Keys are all encoder states

        # Transpose to make [batch x hidden x sentence length]
        decoder_output = decoder_output.transpose(0, 1) # [b x hidden x len]
        encoder_output = encoder_output.transpose(0, 1)

        # for each query / key pair, calculate dot product 
        pdb.set_trace()
        assert(decoder_output.size(1) % 16 == 0)

        # [b x hidden x len] -> [b x len1 x hidden ][b x hidden x len2] -> [b x len1 x le2n]
        attn = torch.bmm(decoder_output.transpose(1,2), encoder_output) 

        # Normalize with softmax
        attn = nn.softmax(attn, dim=2)
        # Output


class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size, n_layers, dropout_p):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size, n_layers, dropout_p)
        self.decoder = DecoderRNN(hidden_size, output_vocab_size, n_layers, dropout_p)


    def forward(self, source, target):
        if USE_CUDA: 
            source = source.cuda()
            target = target.cuda()
        # Encode
        output_encoder, hidden_encoder = self.encoder(source)

        # Decode
        output_decoder, hidden_decoder = self.decoder(target, hidden_encoder)

        # Attend

        # Predict
        return output_decoder
