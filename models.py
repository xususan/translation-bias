import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pdb


# Pytorch 0.4.1 and above

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
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input))
        output, hidden = self.rnn(output, hidden)
        return output, hidden

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, decoder_output, encoder_output):
        # Query is decoder state
        # Keys are all encoder states

        # Transpose to make [batch x hidden x sentence length]
        decoder_output = decoder_output.transpose(0, 1) # [b x hidden x len]
        encoder_output = encoder_output.transpose(0, 1)

        # for each query / key pair, calculate dot product 
        assert(decoder_output.size(1) % 16 == 0)

        # [b x hidden x len] -> [b x outputlen x hidden ][b x hidden x inputlen] -> [b x outputlen x inputlen]
        attn = torch.bmm(decoder_output.transpose(1,2), encoder_output) 

        # Normalize with softmax
        attn = F.softmax(attn, dim=2) # [b x len1 x len2]
        # Output [b x trglen x srclen]
        return attn.transpose(1, 2) # b x srclen x trglen

class AttnDecoderRNN(nn.Module):
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
        self.attn = Attention()

    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input))
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size, n_layers, dropout_p):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size, n_layers, dropout_p)
        self.decoder = AttnDecoderRNN(hidden_size, output_vocab_size, n_layers, dropout_p)
        self.out = nn.Linear(2 * hidden_size, output_size)


    def forward(self, source, target):
        if USE_CUDA: 
            source = source.cuda()
            target = target.cuda()
        # Encode
        encoder_output, encoder_hidden = self.encoder(source)

        # Decode
        decoder_output, decoder_hidden = self.decoder(target, encoder_hidden)

        # Attend
        weights = self.attn(decoder_output, encoder_output) #[b x srclen x trglen]       
        # [b x trglen x srclen] * [b x srclen x hidden] = [b x trglen x hidden]
        context = torch.bmm(weights.tranpose(1,2), encoder_output.transpose(1,2))

        decoder_with_context = torch.cat((decoder_output, context), dim=2)
        return self.out(decoder_output)
