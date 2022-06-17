# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file biggan library
# which was released under the MIT License.
#
# Source:
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/BigGAN.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_biggan). The modifications
# to this file are subject to the same MIT License.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from functools import partial
from torch.autograd import Variable, grad
from .layers import *
from src.architecture.base import Generator


def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[64] = {'in_channels': [ch * item for item in [16, 16, 8, 4]],
                'out_channels': [ch * item for item in [16, 8, 4, 2]],
                'upsample': [True] * 4,
                'resolution': [8, 16, 32, 64],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 7)}}
    arch[32] = {'in_channels': [ch * item for item in [4, 4, 4]],
                'out_channels': [ch * item for item in [4, 4, 4]],
                'upsample': [True] * 3,
                'resolution': [8, 16, 32],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 6)}}
    arch[28] = {'in_channels': [ch * item for item in [4, 4, 4]],
                'out_channels': [ch * item for item in [4, 4, 4]],
                'upsample': [True] * 3,
                'resolution': [8, 16, 32],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 6)}}
    return arch


class BigGan(nn.Module):
    def __init__(self, G_ch=64, latent_dim=128, bottom_width=4, resolution=128,
                 G_kernel_size=3, G_attn='64',
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_shared=True, shared_dim=0, hier=False,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                 BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                 G_init='ortho', skip_init=False, no_optim=False,
                 G_param='SN', norm_style='bn', num_vis_examples=50, label_dim=10, embed_multiplier=10, n_channels=3,
                 class_cond=True,
                 **kwargs):
        super(BigGan, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.latent_dim = latent_dim
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # number of output channels
        self.n_channels = n_channels
        self.img_dim = (self.n_channels, self.resolution, self.resolution)
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # use class conditional
        self.class_cond = class_cond
        # number of classes, for use in categorical conditional generation
        self.n_classes = label_dim
        self.label_dim = label_dim
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else latent_dim
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]
        self.embed_multiplier = embed_multiplier

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.latent_dim // self.num_slots)
            # Recalculate latent dimensionality for even splitting into chunks
            self.latent_dim = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = partial(SNConv2d,
                                      kernel_size=3, padding=1,
                                      num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                      eps=self.SN_eps)
            self.which_linear = partial(SNLinear,
                                        num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                        eps=self.SN_eps)
        else:
            self.which_conv = partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        if self.class_cond:
            self.which_embedding = nn.Embedding
            bn_linear = (partial(self.which_linear, bias=False) if self.G_shared
                         else self.which_embedding)
            self.which_bn = partial(ccbn,
                                    which_linear=bn_linear,
                                    cross_replica=self.cross_replica,
                                    mybn=self.mybn,
                                    input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                                else self.n_classes),
                                    norm_style=self.norm_style,
                                    eps=self.BN_eps)
            # Prepare model
            # If not using shared embeddings, self.shared is just a passthrough
            self.shared = (self.which_embedding(self.n_classes, self.shared_dim) if G_shared
                           else identity())
        else:
            self.which_bn = partial(bn,
                                    eps=self.BN_eps,
                                    cross_replica=self.cross_replica,
                                    mybn=self.mybn
                                    )
            self.shared = identity()
        # First linear layer
        self.linear = self.which_linear(self.latent_dim // self.num_slots,
                                        self.arch['in_channels'][0] * (self.bottom_width ** 2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                                    out_channels=self.arch['out_channels'][index],
                                    which_conv=self.which_conv,
                                    which_bn=self.which_bn,
                                    activation=self.activation,
                                    upsample=(partial(F.interpolate, scale_factor=2)
                                              if self.arch['upsample'][index] else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(bn(self.arch['out_channels'][-1],
                                             cross_replica=self.cross_replica,
                                             mybn=self.mybn),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], self.n_channels))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()
        self.z = torch.rand(num_vis_examples, latent_dim)
        self.label = torch.randint(label_dim, (num_vis_examples,))
        self.num_vis_examples = num_vis_examples
        # Set up optimizer
        # If this is an EMA copy, no need for an optim, so just return now
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            print('Using fp16 adam in G...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0,
                                      eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0,
                                    eps=self.adam_eps)

        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        output = torch.tanh(self.output_layer(h))

        return output

    def sample_labelled(self, z, label):
        # unconditioned samples
        if self.class_cond:
            gen_img = self(z, self.shared(label))
            gen_label = label
        else:
            gen_img = self(z, None)
            gen_label = label
        return gen_img, gen_label

    def sample_labelled_fixed(self):
        return self.sample_labelled(self.z, self.label)

    @property
    def num_trainable_params(self):
        return np.sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    @staticmethod
    def name():
        return 'Convolutional'

    @staticmethod
    def tag():
        return 'cond_biggan'

    def to(self, device):
        self.z = self.z.to(device)
        self.label = self.label.to(device)
        return super(BigGan, self).to(device)


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64', in_channel=3, ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels': [in_channel] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels': [in_channel] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels': [in_channel] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels': [in_channel] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    arch['32fe'] = {'in_channels': [in_channel] + [item * ch for item in [1, 2]],
                    'out_channels': [item * ch for item in [1, 2, 4]],
                    'downsample': [True, True, True],
                    'resolution': [16, 8, 4],
                    'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                                  for i in range(2, 6)}}
    arch['28fe'] = {'in_channels': [in_channel] + [item * ch for item in [4, ]],
                    'out_channels': [item * ch for item in [4, 4, ]],
                    'downsample': [True, True],
                    'resolution': [14, 7],
                    'attention': {14: False, 7: False}
                    }
    return arch


class Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                 D_kernel_size=3, D_attn='64', label_dim=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                 SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                 D_init='ortho', skip_init=False, D_param='SN', fe_mode=True, in_channel=3, class_cond=True, **kwargs):
        super(Discriminator, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # use class conditioning
        self.class_cond = class_cond
        # Number of classes
        self.n_classes = label_dim
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        if not fe_mode:
            self.arch = D_arch(self.ch, self.attention, in_channel)[resolution]
        else:
            self.arch = D_arch(self.ch, self.attention, in_channel)[str(resolution) + 'fe']
        self.fe_only = fe_mode
        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = partial(SNConv2d,
                                      kernel_size=3, padding=1,
                                      num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                      eps=self.SN_eps)
            self.which_linear = partial(SNLinear,
                                        num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                        eps=self.SN_eps)
            self.which_embedding = partial(SNEmbedding,
                                           num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                           eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                    out_channels=self.arch['out_channels'][index],
                                    which_conv=self.which_conv,
                                    wide=self.D_wide,
                                    activation=self.activation,
                                    preactivation=(index > 0),
                                    downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index],
                                              self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        if not self.fe_only:
            self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
            # Embedding for projection discrimination
            self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
        else:
            self.out_conv = nn.Conv2d(self.arch['out_channels'][-1], output_dim, 1, 1, bias=False)

        # Initialize weights
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print('Using fp16 adam in D...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        # print('Param count for D''s initialized parameters: %d' % self.param_count)

    def get_penalty(self, x_true, x_gen, label):
        x_true = x_true.view_as(x_gen).cuda()
        alpha = torch.rand((len(x_true),) + (1,) * (x_true.dim() - 1))
        if x_true.is_cuda:
            alpha = alpha.cuda(x_true.get_device())
        x_penalty = Variable(alpha * x_true + (1 - alpha) * x_gen, requires_grad=True).cuda()
        p_penalty = self.forward(x_penalty, label)
        gradients = grad(p_penalty, x_penalty, grad_outputs=torch.ones_like(p_penalty).cuda(
            x_true.get_device()) if x_true.is_cuda else torch.ones_like(p_penalty), create_graph=True,
                         retain_graph=True, only_inputs=True)[0]
        penalty = ((gradients.view(len(x_true), -1).norm(2, 1) - 1) ** 2).mean()

        return penalty

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        if not self.fe_only:
            # Apply global sum pooling as in SN-GAN
            h = torch.sum(self.activation(h), [2, 3])
            # Get initial class-unconditional output
            out = self.linear(h)
            # Get projection of final featureset onto class vectors and add to evidence
            out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        else:
            out = self.out_conv(self.activation(h))  # down projection
        return out

    @property
    def num_trainable_params(self):
        return np.sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])


class SmallDiscriminator(nn.Module):
    def __init__(self, out_dim, num_hidden_layers, num_hidden_units, in_channel=3):
        super(SmallDiscriminator, self).__init__()
        # self.label_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        # self.proj_y = proj_y
        self.layers = []
        for i in range(num_hidden_layers):
            if i == 0:
                self.layers.append(nn.Conv2d(in_channel, num_hidden_units, 3, 1, 1, bias=False))
            else:
                self.layers.append(nn.Conv2d(num_hidden_units, num_hidden_units, 3, 1, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(num_hidden_units))
            self.layers.append(nn.ELU())
            # if i >= 1:
            self.layers.append(nn.MaxPool2d(2, 2))

        self.net = nn.Sequential(*self.layers)
        self.fe_layer = nn.Sequential(nn.Conv2d(num_hidden_units, out_dim, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(out_dim),
                                      nn.ReLU(),
                                      )
        # if proj_y:
        #    self.embed = nn.Embedding(label_dim, out_dim)

    @property
    def num_trainable_params(self):
        return np.sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    def forward(self, x, y=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.net(x)
        fe = self.fe_layer(h)
        # if self.proj_y:
        #    assert y is not None
        #    fe = fe + torch.sum(self.embed(y) * fe, 1, keepdim=True)
        return fe

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                init.orthogonal_(module.weight)
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        # print('Param count for D''s initialized parameters: %d' % self.param_count)

    def get_penalty(self, x_true, x_gen, x_label):
        x_true = x_true.view_as(x_gen).cuda()
        alpha = torch.rand((len(x_true),) + (1,) * (x_true.dim() - 1))
        if x_true.is_cuda:
            alpha = alpha.cuda(x_true.get_device())
        x_penalty = Variable(alpha * x_true + (1 - alpha) * x_gen, requires_grad=True).cuda()
        p_penalty = self.forward(x_penalty, x_label)
        gradients = grad(p_penalty, x_penalty, grad_outputs=torch.ones_like(p_penalty).cuda(
            x_true.get_device()) if x_true.is_cuda else torch.ones_like(p_penalty), create_graph=True,
                         retain_graph=True, only_inputs=True)[0]
        penalty = ((gradients.view(len(x_true), -1).norm(2, 1) - 1) ** 2).mean()
        return penalty
