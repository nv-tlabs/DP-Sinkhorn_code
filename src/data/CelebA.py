# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
from torchvision.datasets import CelebA

class MyCelebA(CelebA):
    def __init__(self, root, split="train", transform=None,
                 target_transform=None, download=False):
        super(MyCelebA, self).__init__(root, split, "attr", transform, target_transform, download)

    def __getitem__(self, item):
        img, label = super(MyCelebA, self).__getitem__(item)
        code = label[20]
        return img, code
