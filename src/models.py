import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import *
from src.loader import DEFAULT_TOKEN_TO_INDEX
from src.utils import cud


# IMG_H, IMG_W = 60, 400
class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        # input shape: (batch_size, 1, IMG_H, IMG_W)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # , padding=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # , padding=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # , padding=0),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # , padding=0),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # (batch_size, 512, IMG_H/2/2/2, IMG_W/2/2/2)

    def forward(self, x):
        return self.layers(x)


class EncoderRNN(nn.Module):
    def __init__(self, input_d, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_d = input_d
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_d, hidden_size=hidden_size, num_layers=1, dropout=0, bidirectional=True, batch_first=True)

    def forward(self, input):
        return self.gru(input)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, max_seq_len, embedding_dim):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)

        # self.attn_coefficients = nn.Linear(hidden_size + encoder_hidden_size, encoder_max_recursion)

        self.gru = nn.GRU(input_size=embedding_dim+hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden, n, formulas, encoder_outputs):
        if self.training:  # teacher forcing
            return self.training_forward(hidden, n, formulas, encoder_outputs)
        else:
            return self.test_forward(hidden, n, encoder_outputs)

    @staticmethod
    def get_attn_context(encoder_outputs, hidden):
        attn_coefs = F.softmax(
            torch.bmm(encoder_outputs, hidden.permute(1, 2, 0)).squeeze(2), dim=1)  # batch, seq_len (50)
        return torch.bmm(attn_coefs.unsqueeze(1), encoder_outputs).squeeze(1)  # batch, hidden_size

    # We can use packed inputs:
    # packed = nn.utils.rnn.pack_padded_sequence(emb, formula_lengths, batch_first=False, enforce_sorted=False)
    # packed_output, _ = self.gru(packed, hidden)
    # output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False, total_length=self.max_seq_len)
    def training_forward(self, hidden, n, formulas, encoder_outputs):
        emb = self.embedding(formulas)  # batch, seq_len, emb_dim
        output = cud(CUDA, torch.zeros((n, self.max_seq_len, self.hidden_size)))

        for i in range(0, self.max_seq_len):
            context = DecoderRNN.get_attn_context(encoder_outputs, hidden)
            concatenated = torch.cat((emb[:, i, :], context), dim=1).unsqueeze(1)  # batch, 1, emb_dim (80) + hidden_size (200)
            gru_output, hidden = self.gru(concatenated, hidden)
            output[:, i, :] = gru_output.squeeze(1)

        return self.linear(output)  # batch, seq_len, vocab_size

    def test_forward(self, hidden, n, encoder_outputs):
        generated = cud(CUDA, torch.zeros((n, self.max_seq_len)))
        prev = cud(CUDA, torch.tensor(n * [DEFAULT_TOKEN_TO_INDEX['<SOS>']]))
        generated[:, 0] = prev
        for i in range(1, self.max_seq_len):
            emb = self.embedding(prev)  # batch * emb_dim
            context = DecoderRNN.get_attn_context(encoder_outputs, hidden)
            concatenated = torch.cat((emb, context), dim=1).unsqueeze(1)  # batch, 1, emb_dim (80) + hidden_size (200)
            gru_output, hidden = self.gru(concatenated, hidden)
            reshaped_output = gru_output.squeeze(1)  # batch, hidden_size
            linear_output = self.linear(reshaped_output)  # batch, vocab_size
            new_tokens = cud(CUDA, torch.distributions.categorical.Categorical(logits=linear_output).sample())
            generated[:, i] = new_tokens
            prev = new_tokens
        return generated


FEATURE_MAP_H, FEATURE_MAP_W, FEATURE_MAP_D = 7, 50, 512


class Combined(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super(Combined, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.conv = Conv()
        self.dropout = nn.Dropout()
        self.encoder = EncoderRNN(FEATURE_MAP_H * FEATURE_MAP_D, HIDDEN_SIZE)
        # Another dropout
        self.decoder = DecoderRNN(2 * HIDDEN_SIZE, vocab_size, max_seq_len, EMBEDDING_DIM)

    def forward(self, images, formulas=None):  # remember to call model.train()
        conved = self.conv(images)  # batch, D, H, W
        flattened_conv = conved.permute(0, 3, 1, 2).view(-1, FEATURE_MAP_W, FEATURE_MAP_H * FEATURE_MAP_D)  # batch, seq_len, feature_dim

        dropped_out_conv = self.dropout(flattened_conv)

        encoder_outputs, hidden = self.encoder(dropped_out_conv)  # batch, seq_len (50), num_dir (2) * hidden_size | directions (2), batch, hidden_size
        concatenated_h = hidden.permute(1, 0, 2).contiguous().view(-1, 2 * HIDDEN_SIZE).unsqueeze(0)  # 1, batch, 2 *hidden_size

        dropped_out_h = self.dropout(concatenated_h)

        n = images.size()[0]
        return self.decoder(dropped_out_h, n, formulas, encoder_outputs)  # train output is different from test output
