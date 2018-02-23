import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# from attention import Attention
from .attention import Attention


class VecDecoder(nn.Module):
    def __init__(self, decoder_config):
        super(VecDecoder, self).__init__()
        self.input_dim = decoder_config.input_dim
        self.hidden_dim = decoder_config.hidden_dim
        self.output_dim = decoder_config.output_dim
        self.max_length = decoder_config.max_length
        self.use_attention = decoder_config.use_attention
        self.bidirectional = decoder_config.bidirectional
        self.num_layers = decoder_config.num_layers
        self.batch_first = decoder_config.batch_first
        self.num_directions = 1 + int(self.bidirectional)

        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, batch_first=self.batch_first,
                          dropout=decoder_config.dropout, bidirectional=self.bidirectional, num_layers=self.num_layers)
        self.output_input_bridge = nn.Linear(
            self.hidden_dim * self.num_directions, self.input_dim)
        self.hidden_output_bridge = nn.Linear(
            self.hidden_dim * self.num_directions, self.output_dim)
        self.attention = Attention(
            self.num_directions * self.hidden_dim)

    def step(self, decoder_input, decoder_hidden, encoder_outputs):
        batch_size = decoder_input.size(0)
        output_size = decoder_input.size(1)
        # if not self.training:
        #     print('step----Decoder input:', decoder_input.size())
        #     print('step----Decoder hidden:', decoder_hidden.size())
        output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
#         print('step----Decoder hidden:', decoder_hidden.size())
        # output = decoder_hidden
        # output = torch.mean(decoder_hidden, dim=0)
        # output = output.view(batch_size, 1, -1)
        # print('step----Decoder hidden:', output.size())
        # print('step----Encoder outputs:', encoder_outputs.size())
        attn_wts = None
        if self.use_attention:
            output, attn_wts = self.attention(output, encoder_outputs)
#             print('step----Attn mixed vector:', output.size())
#         print('step----Attention weights:', attn_wts.size())
        output = output.contiguous()
#         print('step----Output vector:', output.size())
#         print('-----------------------------------------')
        return output, decoder_hidden, attn_wts

    def forward(self, encoder_hidden, lengths, encoder_outputs=None, targets=None):
        batch_size = encoder_hidden.size()[1]
        decoder_input = Variable(torch.zeros(batch_size, 1, self.input_dim).cuda() + 0.1)
        if encoder_hidden.is_cuda:
            decoder_input.cuda()
        outputs = []
#         print('Encoder hidden:', encoder_hidden.size())
        decoder_hidden = encoder_hidden
        teacher_forcing = True
#         print('Decoder hidden:', decoder_hidden.size())
        if targets is not None:
            timesteps = min(self.max_length, targets.size()[1])
        else:
            timesteps = self.max_length
        for i in range(timesteps):
            decoder_output, decoder_hidden, attn_wts = self.step(
                decoder_input, decoder_hidden, encoder_outputs)
            # print(decoder_output.size())
            # print('')
            if teacher_forcing and self.training and (targets is not None):
                decoder_input = targets[:, i, :]
                decoder_input = decoder_input.unsqueeze(1)
                # print(targets[0, i, :])
                # print(targets[0, i, :])
                # exit()
                # print(decoder_input)
            else:
                decoder_input = decoder_output
#             decoder_hidden = decoder_input
            # print('Dense output:', decoder_input.size())
            outputs.append(decoder_output.squeeze(1))

#         print('<<<<<<************************************************************>>>>>>')
        # print(len(outputs))
        outputs = torch.stack(outputs).transpose_(0, 1)
        # print(outputs.size())
        return outputs, decoder_hidden
