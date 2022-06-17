# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .base import Generator

def identity(x):
    return x

class ConvCondGenerator(Generator):
    def __init__(self, latent_dim, img_dim, label_dim, embed_multiplier, num_vis_examples, embed_dim, hidden_dim, fs='tttn', class_cond = True):
        super(ConvCondGenerator, self).__init__(latent_dim, img_dim, label_dim, embed_multiplier, num_vis_examples)

        assert len(fs) == 4
        self.acfs = []
        for f in fs:
            if f == 'n':
                self.acfs.append(identity)
            elif f == 't':
                self.acfs.append(F.tanh)
            elif f == 'l':
                self.acfs.append(partial(F.leaky_relu, negative_slope=0.2))
            elif f == 's':
                self.acfs.append(F.sigmoid)
            elif f == 'e':
                self.acfs.append(partial(F.elu, alpha=0.2))

        self.class_cond = class_cond
        if self.class_cond:
            self.embed_label = nn.Embedding(label_dim, embed_dim)
            self.conv1 = nn.ConvTranspose2d(latent_dim+embed_dim, hidden_dim * 4, 7, 1, 0, bias=False)
            #self.conv1 = nn.ConvTranspose2d(latent_dim+embed_dim, hidden_dim * 4, 7, 1, 0, bias=True)
        else:
            self.conv1 = nn.ConvTranspose2d(latent_dim, hidden_dim * 4, 7, 1, 0, bias=True)

        self.norm1 = nn.BatchNorm2d(hidden_dim*4, affine=True)
        #self.norm1 = nn.BatchNorm2d(hidden_dim*4, affine=False)

        # hidden_dim*4 x 7 x 7 ksp

        self.conv2 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias = False)
        self.norm2 = nn.BatchNorm2d(hidden_dim * 2)
        # hidden_dim*2 x 14 x 14

        self.conv3 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias = False)
        self.norm3 = nn.BatchNorm2d(hidden_dim)
        # hidden_dim x 28 x 28

        self.conv4 = nn.Conv2d(hidden_dim, img_dim[0], 3, 1, 1, bias = True)
        # 1 x 28 x 28

    # forward method
    def forward(self, z, label):
        if self.class_cond:
            label_embedding = self.embed_label(label)
            z0 = torch.cat([z, label_embedding], dim=1)
        else:
            z0 = z
        z0 = z0.view(z0.shape + (1,1))
        # z x 1 x 1

        z1 = self.acfs[0](self.norm1(self.conv1(z0)))
        # 4h x 7 x 7

        z2 = self.acfs[1](self.norm2(self.conv2(z1)))
        # 2h x 14 x 14

        z3 = self.acfs[2](self.norm3(self.conv3(z2)))
        # h x 28 x 28

        z4 = self.acfs[3](self.conv4(z3))
        # 784

        return z4

    def sample_labelled(self, z, label):
        out = self(z, label)
        gen_img = out
        gen_label = label
        return gen_img, gen_label

    @staticmethod
    def name():
        return 'Convolutional Conditional'

    @staticmethod
    def tag():
        return 'conv_cond'

class ConvCondSqGenerator(Generator):
    def __init__(self, latent_dim, img_dim, label_dim, embed_multiplier, num_vis_examples, embed_dim, hidden_dim, fs='lllt', class_cond=True):
        super(ConvCondSqGenerator, self).__init__(latent_dim, img_dim, label_dim, embed_multiplier, num_vis_examples)
        assert len(fs) == 4
        self.acfs = []
        for f in fs:
            if f == 'n':
                self.acfs.append(identity)
            elif f == 't':
                self.acfs.append(F.tanh)
            elif f == 'l':
                self.acfs.append(partial(F.leaky_relu, negative_slope = 0.2))
            elif f == 's':
                self.acfs.append(F.sigmoid)
            elif f == 'e':
                self.acfs.append(partial(F.elu, alpha=0.2))

        self.class_cond = class_cond
        if self.class_cond:
            self.embed_label = nn.Embedding(label_dim, embed_dim)
            self.conv1 = nn.ConvTranspose2d(latent_dim + embed_dim, hidden_dim * 4, 4, 1, 0, bias=True)
        else:
            self.conv1 = nn.ConvTranspose2d(latent_dim, hidden_dim * 4, 4, 1, 0, bias=True)

        self.norm1 = nn.BatchNorm2d(hidden_dim*4, affine=False)
        # hidden_dim*4 x 4 x 4

        self.conv2 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias = False)
        self.norm2 = nn.BatchNorm2d(hidden_dim*2)
        # hidden_dim*2 x 8 x 8

        self.conv3 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias = False)
        self.norm3 = nn.BatchNorm2d(hidden_dim)
        # hidden_dim x 16 x 16

        self.conv4 = nn.ConvTranspose2d(hidden_dim, img_dim[0], 4, 2, 1, bias = False)
        # nc x 32 x 32


    # forward method
    def forward(self, z, label):
        label_embedding = self.embed_label(label)
        z0 = torch.cat([z, label_embedding], dim=1)
        z0 = z0.view(z0.shape + (1,1))
        # z x 1 x 1

        z1 = self.acfs[0](self.norm1(self.conv1(z0)))
        # 4h x 4 x 4

        z2 = self.acfs[1](self.norm2(self.conv2(z1)))
        # 2h x 8 x 8

        z3 = self.acfs[2](self.norm3(self.conv3(z2)))
        # h x 16 x 16

        z4 = self.acfs[3](self.conv4(z3))
        # c x 32 x 32

        return z4

    def sample_labelled(self, z, label):
        out = self(z, label)
        gen_img = out
        gen_label = label
        return gen_img, gen_label

    @staticmethod
    def name():
        return 'Convolutional Conditional Square'

    @staticmethod
    def tag():
        return 'conv_cond_sq'