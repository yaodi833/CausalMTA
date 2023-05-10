import torch
import torch.nn as nn

class LSTM_Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        LSTM_hidden_layer_depth = 1,
        dropout_rate = 0.2
    ):
        super(LSTM_Encoder, self).__init__()
        self.model = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = LSTM_hidden_layer_depth,
            batch_first = True,
            dropout = dropout_rate
        )
        
    def forward(self, x):
        _, (h_end, c_end) = self.model(x)
        # h_end = h_end[:, -1, :]
        return h_end[-1]


class LSTM_Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        batch_size,
        LSTM_hidden_layer_depth = 1,
        dropout_rate = 0.2,
        device = "cuda:0"
    ):
        super(LSTM_Decoder, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.LSTM_hidden_layer_depth = LSTM_hidden_layer_depth
        self.device = device

        self.model = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = LSTM_hidden_layer_depth,
            batch_first = True,
            dropout = dropout_rate
        )

    def forward(self, h_state, seq_len):
        decoder_inputs = torch.zeros(h_state.shape[0], seq_len, self.input_dim).to(self.device)
        c_0 = torch.zeros(self.LSTM_hidden_layer_depth, h_state.shape[0], self.hidden_dim).to(self.device)
        h_0 = h_state.repeat(self.LSTM_hidden_layer_depth, 1, 1)

        # print(decoder_inputs.shape)
        # print(c_0.shape)
        # print(h_0.shape)

        decoder_output, _ = self.model(decoder_inputs, (h_0, c_0))
        return decoder_output

