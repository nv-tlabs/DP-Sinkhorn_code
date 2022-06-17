# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the TensorFlow Privacy
# which was released under the Apache License v2.0 License.
#
# Source:
# https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/compute_dp_sgd_privacy_lib.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_RDP). The modifications
# to this file are subject to the same Apache License.
# ---------------------------------------------------------------
import torch
from torch.optim import Adam
import numpy as np
from .rdp_accountant import get_privacy_spent, compute_rdp_sample_without_replacement, compute_rdp


def add_privacy_args(parser):
    parser.add_argument("--dp", default=False, action='store_true')
    parser.add_argument("--report_bdp", default=False, action='store_true')
    parser.add_argument("--sampling", default='subset', choices=['subset', 'poisson'])
    parser.add_argument("--mechanism", default='img_grad', choices=['img_grad'])

    parser.add_argument("--l2_norm_clip", type=float, default=0.5)
    parser.add_argument("--noise_multiplier", type=float, default=0.7)
    #parser.add_argument("--noise_transport", default=False, action="store_true")
    parser.add_argument("--plan_noise_multiplier", type=float, default=0.3)
    parser.add_argument("--sensitivity_to_l2_norm_clip", default=2, type=float)
    '''
    noise sigma = noise_muliplier * sensitivity 
                = noise_multiplier * sensitivity_to_l2_norm_clip * l2_norm_clip
    '''
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--js_estimate", action='store_true', default=False)

    return parser


gradient_moving_avg = None


def apply_shrinkage(x, args, gnorm_mets):
    global gradient_moving_avg

    if gradient_moving_avg is None:
        gradient_moving_avg = torch.zeros(x.shape).to(x.device)

    d = np.prod(np.array(list(x.shape)))

    shrinkage_factor = 1 - (
                (d - 2) * (args.noise_multiplier * args.l2_norm_clip * args.sensitivity_to_l2_norm_clip) ** 2 / (
                    x - gradient_moving_avg).norm() ** 2)

    gnorm_mets['shrinkage_factor'] = shrinkage_factor.item()

    gamma = 0.9
    gradient_moving_avg = gamma * gradient_moving_avg + (1 - gamma) * x

    return shrinkage_factor * x


def record_img_grads(x, gnorm_mets):
    clean_gnorm = x.norm().item()
    gnorm_mets['gnorm/clean'] = clean_gnorm
    return x

def per_rowclip_img_grads_hook(x, args, gnorm_mets):
    gnorms = x.view(x.shape[0], -1).norm(dim=1)
    gnorm_mets['gnorm/clean'] = x.norm().item()
    clip_coefficients = torch.clamp_max(args.l2_norm_clip/(gnorms + 1e-6), 1.0)
    x = x * clip_coefficients.view(x.shape[0],1,1,1)
    gnorm_mets['gnorm/clipped'] = x.norm().item()
    return x

def clip_img_grads_hook(x, args, gnorm_mets):
    clean_gnorm = x.norm().item()
    clip_coef = min(args.l2_norm_clip / (clean_gnorm + 1e-6), 1.)
    x = x * clip_coef  # clip to clip norm
    return x

def per_row_sanitize_img_grads_hook(x, args, gnorm_mets):
    # for use with discretized sample loss
    gnorms = x.view(x.shape[0], -1).norm(dim=1)
    gnorm_mets['gnorm/clean'] = x.norm().item()
    #print(gnorms)
    clip_coefficients = torch.clamp_max(args.l2_norm_clip/(gnorms + 1e-6), 1.0)
    #print(x.shape)
    #print(clip_coefficients.shape)
    #print(clip_coefficients)
    x = x * clip_coefficients.view(x.shape[0],1,1,1)
    # each real sample at most flips one row of X, so l2-sensitivity is 2c(x) rather than 2C(X)
    noise = (torch.randn(x.shape) * args.l2_norm_clip * args.noise_multiplier * args.sensitivity_to_l2_norm_clip).to(
        x.device)  # add noise to sensitivity 2 fn

    gnorm_mets['gnorm/clipped'] = x.norm().item()
    gnorm_mets['gnorm/noise'] = noise.norm().item()

    x = x + noise

    #x = x * args.l2_norm_clip  # post processing to original size

    if args.js_estimate:
        x = apply_shrinkage(x, args, gnorm_mets)

    return x

def sanitize_img_grads_hook(x, args, gnorm_mets):
    clean_gnorm = x.norm().item()
    clip_coef = min(args.l2_norm_clip / (clean_gnorm + 1e-6), 1.)
    x = x * clip_coef  # clip to clip norm
    #x = x / args.l2_norm_clip  # move to 1 norm bound
    # print (x.norm().item())
    # print (x.shape)

    noise = (torch.randn(x.shape) * args.l2_norm_clip * args.noise_multiplier * args.sensitivity_to_l2_norm_clip).to(
        x.device)  # add noise to sensitivity 2 fn

    gnorm_mets['gnorm/clean'] = clean_gnorm
    gnorm_mets['gnorm/noise'] = noise.norm().item()

    x = x + noise

    #x = x * args.l2_norm_clip  # post processing to original size

    if args.js_estimate:
        x = apply_shrinkage(x, args, gnorm_mets)

    return x

def calc_two_part_epsilon(batch_size, noise_multiplier1, noise_multiplier2, delta, N, iterations, sampling):
    '''

    if sampling is 'subset':
    - N examples are broken up into 'records' which are groups of size 'args.batch_size'
    - on each iteration:
        - we sample 1 record, corresponding to sampling without replacement with parameter q = 'args.batch_size / N'
        - clip to norm C, add coordinate wise noise with sigma = noise_multiplier * C * 2
        - release gradient

    if sampling is 'poisson':
    - on each iteration:
        - we draw a subset S from the training data, each example is included with probability 'args.batch_size/N', and compute gradients
        - clip to norm C, add coordinate-wise noise with sigma = noise_multiplier * C * 2
        - release gradient

    - compute RDP privacy loss per iteration (w/ params noise_muliplier, q)
    - multiply by number of iterations
    - find minimum epsilon such that (Ɛ,𝛿)-DP is satisfied
    '''

    q = batch_size / N

    steps = iterations

    if sampling == 'subset':
        orders = ([2, 2.25, 2.5, 3, 3.5, 4, 4.5] +
                  list(range(5, 64)) + [128, 256, 512, 1024, 2048, 4096])
        rdp1 = compute_rdp_sample_without_replacement(q, noise_multiplier1, steps, orders)
        rdp2 = compute_rdp_sample_without_replacement(q, noise_multiplier2, steps, orders)
    elif sampling == 'poisson':
        orders = ([1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 4.5] +
                  list(range(5, 64)) + [128, 256, 512, 1024, 2048, 4096])
        rdp1 = compute_rdp(q, noise_multiplier1, steps, orders)
        rdp2 = compute_rdp(q, noise_multiplier2, steps, orders)
    else:
        raise ValueError("Sampling not recognized %s" % sampling)
    epsilon, _, opt_order = get_privacy_spent(orders, rdp1+rdp2, target_delta=delta)

    if opt_order == max(orders) or opt_order == min(orders):
        print('| The privacy estimate is likely to be improved by expanding the set of orders.')
        print('| opt_order : {}'.format(opt_order))

    return epsilon

def calc_epsilon(batch_size, noise_multiplier, delta, N, iterations, sampling):
    '''

    if sampling is 'subset':
    - N examples are broken up into 'records' which are groups of size 'args.batch_size'
    - on each iteration:
        - we sample 1 record, corresponding to sampling without replacement with parameter q = 'args.batch_size / N'
        - clip to norm C, add coordinate wise noise with sigma = noise_multiplier * C * 2
        - release gradient

    if sampling is 'poisson':
    - on each iteration:
        - we draw a subset S from the training data, each example is included with probability 'args.batch_size/N', and compute gradients
        - clip to norm C, add coordinate-wise noise with sigma = noise_multiplier * C * 2
        - release gradient

    - compute RDP privacy loss per iteration (w/ params noise_muliplier, q)
    - multiply by number of iterations
    - find minimum epsilon such that (Ɛ,𝛿)-DP is satisfied
    '''

    q = batch_size / N
    noise_multiplier = noise_multiplier
    steps = iterations

    if sampling == 'subset':
        orders = ([2, 2.25, 2.5, 3, 3.5, 4, 4.5] +
                  list(range(5, 64)) + [128, 256, 512, 1024, 2048, 4096])
        rdp = compute_rdp_sample_without_replacement(q, noise_multiplier, steps, orders)

    elif sampling == 'poisson':
        orders = ([1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 4.5] +
                  list(range(5, 64)) + [128, 256, 512, 1024, 2048, 4096])
        rdp = compute_rdp(q, noise_multiplier, steps, orders)

    epsilon, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

    if opt_order == max(orders) or opt_order == min(orders):
        print('| The privacy estimate is likely to be improved by expanding the set of orders.')
        print('| opt_order : {}'.format(opt_order))

    return epsilon


def make_dp_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier

        def step(self, *args, **kwargs):

            # compute clip_coef
            clean_gnorm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        clean_gnorm += param.grad.data.norm(2).item() ** 2.
            clean_gnorm = clean_gnorm ** .5
            clip_coef = min(self.l2_norm_clip / (clean_gnorm + 1e-6), 1.)

            # clip and add noise
            noise_gnorm = 0
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        param.grad.data.mul_(clip_coef)
                        grad_noise = self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data)
                        noise_gnorm += grad_noise.norm(2).item() ** 2.
                        param.grad.data.add_(grad_noise)
            noise_gnorm = noise_gnorm ** .5

            super(DPOptimizerClass, self).step(*args, **kwargs)

            return {
                'clean_gnorm': clean_gnorm,
                'noise_gnorm': noise_gnorm
            }

    return DPOptimizerClass


DPAdam = make_dp_optimizer_class(Adam)