# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

from abc import abstractmethod

class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim, label_dim, embed_multiplier, num_vis_examples):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_dim = img_dim
        self.label_dim = label_dim
        self.embed_multiplier = embed_multiplier
        self.num_vis_examples = num_vis_examples

        self.z = torch.rand(num_vis_examples, latent_dim)
        self.label = torch.randint(label_dim, (num_vis_examples,))

    @property
    def num_trainable_params(self):
        return np.sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    @staticmethod
    @abstractmethod
    def name():
        pass

    @staticmethod
    @abstractmethod
    def tag():
        pass

    @abstractmethod
    def sample_labelled(self, z, label):
        pass

    def sample_labelled_fixed(self):
        return self.sample_labelled(self.z, self.label)

    def to(self, device):
        self.z = self.z.to(device)
        self.label = self.label.to(device)
        return super(Generator, self).to(device)