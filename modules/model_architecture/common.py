
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math

import torch
from torch import nn
from torch.nn.utils import weight_norm as wn
import torch.nn.functional as F
from torch.autograd import Variable
logger = logging.getLogger(__name__)
import numpy as np

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaPreTrainedModel, RobertaOutput, RobertaSelfOutput, RobertaIntermediate

class RobertaSelfEncoder(nn.Module):
    def __init__(self, config):
        super(RobertaSelfEncoder, self).__init__()
        layer = RobertaLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class RobertaCoAttention(nn.Module):
    def __init__(self, config):
        super(RobertaCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    
    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_attentions=False):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply the attention mask
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        if output_attentions:
            return context_layer, attention_probs
        return context_layer


class RobertaCrossAttention(nn.Module):
    def __init__(self, config):
        super(RobertaCrossAttention, self).__init__()
        self.self = RobertaCoAttention(config)
        self.output = RobertaSelfOutput(config)  # You'll need to define this or import it

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask, output_attentions=False):
        if output_attentions:
            s1_cross_output, attention_probs = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask, output_attentions=True)
        else:
            s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask, output_attentions=False)
            attention_probs = None
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        if output_attentions:
            return attention_output, attention_probs
        return attention_output


class RobertaCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(RobertaCrossAttentionLayer, self).__init__()
        self.attention = RobertaCrossAttention(config)
        self.intermediate = RobertaIntermediate(config)  # You'll need to define this or import it
        self.output = RobertaOutput(config)  # You'll need to define this or import it

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_attentions=False):
        if output_attentions:
            attention_output, attention_probs = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask, output_attentions=True)
        else:
            attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask, output_attentions=False)
            attention_probs = None
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return layer_output, attention_probs
        return layer_output

class RobertaCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(RobertaCrossEncoder, self).__init__()
        layer = RobertaCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True, output_attentions=False):
        all_encoder_layers = []
        all_attentions = []
        hidden_states = s1_hidden_states
        
        for layer_module in self.layer:
            if output_attentions:
                layer_output, layer_attentions = layer_module(hidden_states, s2_hidden_states, s2_attention_mask, output_attentions=True)
                all_attentions.append(layer_attentions)
            else:
                layer_output = layer_module(hidden_states, s2_hidden_states, s2_attention_mask, output_attentions=False)
            hidden_states = layer_output
            if output_all_encoded_layers:
                all_encoder_layers.append(layer_output)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(layer_output)
        
        if output_attentions:
            return all_encoder_layers, all_attentions
        return all_encoder_layers

class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1),
                 shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                            stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):,
              :(xs[3] - self.filter_size[1] + 1)]
        return x
    
def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when righshifting, the last column is removed
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
                 int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]

class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        # TODO : try with original ordering
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)

class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1),
                 shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(
            num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad = nn.ZeroPad2d((int((filter_size[1] - 1) / 2),  # pad left
                                 int((filter_size[1] - 1) / 2),  # pad right
                                 filter_size[0] - 1,            # pad top
                                 0))                           # pad down

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down:
            self.down_shift = lambda x: down_shift(
                x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x

class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1),
                 shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(
            num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right:
            self.right_shift = lambda x: right_shift(
                x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))

class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op: down_shifted_conv2d, nonlinearity=concat_elu, skip_connection=0, h_dim=768):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(
            2 * num_filters, num_filters)  # cuz of concat elu

        if skip_connection != 0:
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)

        self.hw = nn.Parameter(torch.empty(h_dim, num_filters * 2).uniform_(-0.1, 0.1))
        self.num_filters = num_filters

    def forward(self, og_x, a=None, h=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None:
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)

        if h is not None:
            # print('h', h.shape)
            # print('hw', self.hw.shape)
            # print('x', x.shape)
            # print('matmul', torch.matmul(h, self.hw).shape)
            # print('hw_view', torch.matmul(h, self.hw).view(og_x.shape[0], 2*self.num_filters, 1, 1).shape)
            h = torch.matmul(h, self.hw).view(
                og_x.shape[0], 2*self.num_filters, 1, 1)
            h = h.repeat(1, 1, x.shape[-1], x.shape[-1])
            x += h

        a, b = torch.chunk(x, 2, dim=1)
        # print('a', a.shape)
        # print('b', b.shape)
        c3 = a * torch.sigmoid(b)
        # print('og_x', og_x.shape)
        # print('c3', c3.shape)
        # print('og_x+c3', (og_x+c3).shape)
        return og_x + c3

class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, h_dim=768):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=1, h_dim=h_dim)
                                       for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2, h_dim=h_dim)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list, h=None):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop(), h=h)
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1), h=h)

        return u, ul


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, h_dim=768):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=0, h_dim=h_dim)
                                       for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1, h_dim=h_dim)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, h=None):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=None, h=h)
            ul = self.ul_stream[i](ul, a=u, h=h)
            u_list += [u]
            ul_list += [ul]

        return u_list, ul_list

class ImageDecoder(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3, h_dim=768):
        super(ImageDecoder, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception(
                'right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                             self.resnet_nonlinearity, h_dim=h_dim) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                         self.resnet_nonlinearity, h_dim=h_dim) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in range(2)])

        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2, 3),
                                          shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                                          filter_size=(1, 3), shift_output_down=True),
                                      down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                                                filter_size=(2, 1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, h=None, sample=False):
        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(
                xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding
        else:
            if(x.shape != self.init_padding.tolist()):
                xs = [int(y) for y in x.size()]
                padding = Variable(torch.ones(
                    xs[0], 1, xs[2], xs[3]), requires_grad=False)
                self.init_padding = padding.cuda() if x.is_cuda else padding
        if sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(
                xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1], h=h)
            u_list += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list, h=h)

            # upscale (only twice)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(
        xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs +
                                               [nr_mix]).cuda(), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(
        cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + \
        (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.mean(log_sum_exp(log_probs))


def LOSS_TI(real, fake): return discretized_mix_logistic_loss(real, fake)

def calculate_multi_loss(loss_it, loss_ti, loss_task, args):
    loss_final = args.alpha * loss_it / (args.max_seq_length) \
        + args.beta * loss_ti / (args.ti_crop_size * args.ti_crop_size) \
        + args.sigma * loss_task / (args.max_seq_length)
    return loss_final