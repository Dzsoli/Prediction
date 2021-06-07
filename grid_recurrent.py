from model import *


class Encoder_recurrent_grid2d(nn.Module):
    def __init__(self, input_size, hidden, layers, bidir):
        super(Encoder_recurrent_grid2d, self).__init__()

        self.rnn = nn.GRU(input_size, hidden, layers, bidirectional=bidir)
        # input of shape(seq_len, batch, input_size)
        # h_0 of shape(num_layers * num_directions, batch, hidden_size)
        # output of shape(seq_len, batch, num_directions * hidden_size)

        self.grid_encoder = Encoder_Grid2D()
        param = torch.load("../vae2dgrid_2/log_aae_gauss_lindisc_noanneal_refined_model_v3/" +
                   "model_state_dict_aae_gauss_lindisc_noanneal_refined_model_v3")
        enc_param = {}
        for k,v in param.items():
            if "encoder" in k:
                key = k.split("encoder.")[1]
                enc_param[key] = v
        self.grid_encoder.load_state_dict(enc_param)

    def forward(self, x):
        B, C, X, Y, L = x.shape
        z, _ = self.grid_encoder(x.permute((0,4,1,2,3)).reshape((B*L,C,X,Y)))
        z = z.reshape((B,L,4,16)).permute(1,0,2,3).reshape(B,L,64)
        print(z.shape)
        _,(hidden, cell) = self.rnn(z)

        return hidden, cell




model = Encoder_recurrent_grid2d(64,64,1,True)
t = torch.randn((17,1,16,128,12))
print(model(t)[0].shape)