from torch import nn
from .DecoderRNN import DecoderRNN
from .VecEncoder import VecEncoder


class MySeq2Seq(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(MySeq2Seq, self).__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.encoder = VecEncoder(encoder_config)
        self.decoder = DecoderRNN(decoder_config.vocab_size,
                                  decoder_config.max_len,
                                  decoder_config.hidden_size,
                                  decoder_config.sos_id,
                                  decoder_config.eos_id,
                                  decoder_config.num_layers,
                                  'gru',
                                  decoder_config.bidirectional,
                                  decoder_config.input_dropout_p,
                                  decoder_config.dropout_p,
                                  decoder_config.use_attention)

    def forward(self, inputs, lengths=None, targets=None, teacher_forcing_ratio=0.5):
        encoder_outputs, encoder_hidden = self.encoder(inputs, lengths)
        # print(encoder_outputs.size())

        result = self.decoder(inputs=targets,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
