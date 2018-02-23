from torch import nn
from .VecDecoder import VecDecoder
from .VecEncoder import VecEncoder


class VecSeq2Seq(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(VecSeq2Seq, self).__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        # self.bridge_input_dim = encoder_config.hidden_dim * \
        #     (1 + int(encoder_config.bidirectional)) * encoder_config.num_layers
        # self.bridge_output_dim = decoder_config.hidden_dim * \
        #     (1 + int(decoder_config.bidirectional)) * decoder_config.num_layers
        # self.bridge = nn.Linear(self.bridge_input_dim,
        #                         self.bridge_output_dim, bias=False)
        self.encoder = VecEncoder(encoder_config)
        self.decoder = VecDecoder(decoder_config)

    def forward(self, inputs, lengths=None, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs, lengths)
        # print(encoder_outputs.size())
        batch_size = encoder_hidden.size()[1]
        # if self.bridge_input_dim != self.bridge_output_dim:
        #     print('Bridging... ... ...')
        #     encoder_hidden = encoder_hidden.contiguous().view(batch_size, -1)
        #     scaled_encoder_hidden = self.bridge(encoder_hidden)
        #     scaled_encoder_hidden = scaled_encoder_hidden.view(
        #         -1, batch_size, self.decoder_config.hidden_dim)
        # else:
        #     scaled_encoder_hidden = encoder_hidden
        # decoder_outputs, decoder_hidden = self.decoder(
        #     scaled_encoder_hidden, lengths, encoder_outputs)
        # assert self.encoder_config.hidden_dim == self.decoder_config.hidden_dim
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_hidden, lengths, encoder_outputs, targets)
        return decoder_outputs
