from torch import nn


class VecEncoder(nn.Module):
    def __init__(self, encoder_config):
        super(VecEncoder, self).__init__()
        self.input_dim = encoder_config.input_size
        self.hidden_size = encoder_config.hidden_size
        self.batch_first = encoder_config.batch_first
        self.bidirectional = encoder_config.bidirectional
        self.num_layers = encoder_config.num_layers
        self.variable_lengths = encoder_config.variable_lengths
        self.num_directions = 1 + int(self.bidirectional)

        self.rnn = nn.GRU(encoder_config.input_size, encoder_config.hidden_size, batch_first=encoder_config.batch_first,
                          dropout=encoder_config.dropout, bidirectional=encoder_config.bidirectional, num_layers=encoder_config.num_layers)

    def forward(self, inputs, lengths):
        if self.variable_lengths:
            inputs = nn.utils.rnn.pack_padded_sequence(
                inputs, lengths, batch_first=self.batch_first)
        encoder_outputs, encoder_hidden = self.rnn(inputs)
        if self.variable_lengths:
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                encoder_outputs, batch_first=self.batch_first)
        return encoder_outputs, encoder_hidden
