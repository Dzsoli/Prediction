from model import *


class Encoder_recurrent_grid2d(nn.Module):
    def __init__(self, input_size, hidden, layers, bidir):
        super(Encoder_recurrent_grid2d, self).__init__()

        self.rnn = nn.GRU(input_size, hidden, layers, bidirectional=bidir)
        # input of shape(seq_len, batch, input_size)
        # h_0 of shape(num_layers * num_directions, batch, hidden_size)
        # output of shape(seq_len, batch, num_directions * hidden_size)

        self.grid_encoder = Encoder_Grid2D()

    def forward(self, x):
        