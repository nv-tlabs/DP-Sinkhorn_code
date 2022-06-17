# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the pytorch_fid library
# which was released under the Apache License v2.0 License.
#
# Source:
# https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/fid_score.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_fid). The modifications
# to this file are subject to the same Apache License.
# ---------------------------------------------------------------
"""
This file implements functions for computing FID score using pytorch during training
used by trainer.py
"""
import numpy as np
import torch

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
import os, pickle
from .data import fetch_data


# fid -----------------------------------------------
def fid_preproc(img, norm_tuple):
    '''
    @param norm_tuple: (bias, scale) or None
    '''
    if not isinstance(img, torch.Tensor):
        import warnings
        warnings.warn("FID preproc argument is not tensor, converting...")
        img = torch.tensor(img)

    assert img.dim() in (3,4)
    if norm_tuple is not None:
        bias, scale = norm_tuple
        img = (img * scale + bias).clamp(0, 1)
    else:
        img_min = torch.view(img.shape[0],-1).min(dim=1)[0].view(img.shape[0],1,1,1)
        img_max = torch.view(img.shape[0],-1).max(dim=1)[0].view(img.shape[0],1,1,1)

        img = (img - img_min)/ (img_max - img_min + 1e-6).clamp(0,1)
    if len(img.shape)==3:
        img = img.unsqueeze(1)
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)
    return img


def get_activations(img, model, batch_size=50, dims=2048):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    pred_arr = np.empty((len(img), dims))

    for i in tqdm(range(0, len(img), batch_size)):
        start = i
        end = i + batch_size

        batch = img[start:end]

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    return pred_arr


def precompute_fid_activations(dataset_name, datadir, model, device, fid_checkpoint_path):
    (train_data, _), _ = fetch_data(
        dataset_name=dataset_name,
        datadir=datadir,
        as_array=True,
        download=False
    )

    train_img = fid_preproc(torch.tensor(train_data, dtype=torch.float32), (0., 1.)).to(device)

    act = get_activations(train_img, model)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    with open(fid_checkpoint_path, "wb") as fp:
        pickle.dump({'mu': mu, 'sigma': sigma}, fp)

    return mu, sigma


def compute_fid(gen_img, args):
    model = InceptionV3(resize_input=True, normalize_input=True).to(args.device)

    # get mu, sigma of the real data features

    fid_checkpoint_path = os.path.join('pytorch_fid_checkpoints', args.dataset + "_fid.pkl")
    os.makedirs(os.path.dirname(fid_checkpoint_path), exist_ok=True)
    if os.path.exists(fid_checkpoint_path):
        with open(fid_checkpoint_path, 'rb') as fp:
            cdict = pickle.load(fp)
            mu = cdict['mu']
            sigma = cdict['sigma']
    else:
        mu, sigma = precompute_fid_activations(args.dataset,
                                               args.datadir,
                                               model,
                                               args.device,
                                               fid_checkpoint_path)
        with open(fid_checkpoint_path, 'wb') as fp:
            pickle.dump({'mu':mu, 'sigma':sigma}, fp)

    # get mu, sigma of generated image features
    gen_act = get_activations(gen_img, model)
    gen_mu = np.mean(gen_act, axis=0)
    gen_sigma = np.cov(gen_act, rowvar=False)
    fidscore = calculate_frechet_distance(mu, sigma, gen_mu, gen_sigma)

    return fidscore
