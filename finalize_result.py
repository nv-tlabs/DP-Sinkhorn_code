# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import argparse
import os
import numpy as np
import PIL.Image as Image
import math
import time
import tensorflow as tf

import copy
from src.trainer import generate
from src.train_mnist_classifier import torch_evaluate
from src.fid import fid
from src.data import fetch_data
from src.architecture import add_generator_args, build_generator



def tensor_to_arr(t):
    arr = t.detach().cpu().numpy().transpose(0,2,3,1)
    if arr.shape[-1] == 1:
        arr = arr * np.ones((1,1,1,3))
    arr = ((arr * 0.5 + 0.5).clip(0,1) * 255).astype(np.uint8)
    return arr


def arr_to_grid(arr, row_length=8):
    rows = []
    for i in range(math.ceil(arr.shape[0]/row_length)):
        row = np.pad(arr[i*row_length:(i+1)*row_length], pad_width= [(0,0), (0,0), (0,0), (0,0)], constant_values=200)
        actual_row_length = row.shape[0]
        row = np.concatenate([row, 255 * np.ones((row_length - actual_row_length,) + row.shape[1:], dtype=np.uint8)], axis=0)
        row = np.concatenate(row, axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0).repeat(4,axis=0).repeat(4,axis=1)
    return Image.fromarray(grid)


def tensor_to_grid(t, row_length=10):
    return arr_to_grid(tensor_to_arr(t), row_length = row_length)


def sorted_grid(img, label):
    grid = torch.cat([img[label == i][:10] for i in range(10)]).reshape(10,10, 1, 28,28).permute(1,0,2,3,4).reshape(10*10,1,28,28)

    return tensor_to_grid(grid)

def sorted_grid_celeb(img, label):
    grid = torch.cat([img[label == i][:10] for i in range(2)]).reshape(10,2, 3, 32,32).permute(1,0,2,3,4).reshape(10*2,3,32,32)

    return tensor_to_grid(grid)

def get_activations(img, inception_dir, fid_checkpoint_dir=None):
    if (fid_checkpoint_dir is not None and
            (os.path.exists(os.path.join(fid_checkpoint_dir, 'mu.npy')) and
             os.path.exists(os.path.join(fid_checkpoint_dir, 'cov.npy')))):

        mu_path = os.path.join(fid_checkpoint_dir, 'mu.npy')
        cov_path = os.path.join(fid_checkpoint_dir, 'cov.npy')

        print('| computed statistics found at: {}, {}'.format(mu_path, cov_path))

        return np.load(mu_path), np.load(cov_path)

    else:

        png = tensor_to_arr(img)
        inception_dir = fid.check_or_download_inception(inception_dir)
        fid.create_inception_graph(str(inception_dir))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            mu, cov = fid.calculate_activation_statistics(png, sess, verbose=True)

        if fid_checkpoint_dir is not None:
            np.save(os.path.join(fid_checkpoint_dir, 'mu.npy'), mu)
            np.save(os.path.join(fid_checkpoint_dir, 'cov.npy'), cov)

        return mu, cov


def average_checkpoints_load(path, start, k):
    paths = [x for x in os.listdir(path) if x.endswith('.pt')]
    epochs = np.array([int(x.split('_')[0]) if '.' not in x.split('_')[0] else int(x.split('_')[1]) for x in paths])
    order = np.argsort(epochs)
    sorted_epochs = epochs[order]
    start_order_idx = np.where(sorted_epochs == start)[0][0]
    order_idx = order[start_order_idx - k + 1:start_order_idx + 1]

    checkpoint_paths = [os.path.join(path, x) for x in np.array(paths)[order_idx]]

    checkpoint = torch.load(checkpoint_paths[-1])

    print('| averaging {} checkpoints:', len(checkpoint_paths), checkpoint_paths)

    for i in range(k - 1):
        checkpoint_i = torch.load(checkpoint_paths[i])
        for key in checkpoint['state_dict']['g'].keys():
            checkpoint['state_dict']['g'][key] += checkpoint_i['state_dict']['g'][key]

    for key in checkpoint['state_dict']['g'].keys():
        checkpoint['state_dict']['g'][key] /= k

    return checkpoint


def load_checkpoint_for_eval(path, start, k, args, metadata):
    if path.endswith('.pt'):
        checkpoint = torch.load(path)
    else:
        checkpoint = average_checkpoints_load(path, start, k)

    args = copy.deepcopy(args)
    args.__dict__.update(checkpoint['args'])
    args.print_interval = 999999
    args.class_cond = True
    g = build_generator(args, metadata['img_dim'], metadata['label_dim'])

    g.load_state_dict(checkpoint['state_dict']['g'])
    g.to('cuda')
    if g.tag() == 'conv_cond':
        g.norm1.track_running_stats = False
        g.norm1.running_mean = None
        g.norm1.running_var = None

        g.norm2.track_running_stats = False
        g.norm2.running_mean = None
        g.norm2.running_var = None

        g.norm3.track_running_stats = False
        g.norm3.running_mean = None
        g.norm3.running_var = None

    return g, args


def run_eval_pipeline(img, label, val_img, val_label, args):
    if args.dataset =="celeb_32_2":
        grid = sorted_grid_celeb(img, label)
    else:
        grid = sorted_grid(img, label)

    # torch log_reg
    img[img < 0.] = -1.0
    start_time = time.time()
    torch_log_reg_acc, _ = torch_evaluate(img, label, val_img, val_label, None, 'LogReg')
    print('| torch log_reg acc:', torch_log_reg_acc)
    print('| time:', time.time() - start_time)

    # torch mlp
    start_time = time.time()
    torch_mlp_acc, _ = torch_evaluate(img, label, val_img, val_label, None, 'MLP', patience=30)
    print('| torch mlp acc:', torch_mlp_acc)
    print('| time:', time.time() - start_time)

    # torch cnn
    start_time = time.time()
    torch_cnn_acc, _ = torch_evaluate(img, label, val_img, val_label, None, 'CNN', patience=60, optim_choice='SGD')
    print('| torch cnn acc:', torch_cnn_acc)
    print('| time:', time.time() - start_time)

    # fid
    try:
        start_time = time.time()
        mu, cov = get_activations(img, args.inception_dir, None)
        val_mu, val_cov = get_activations(val_img, args.inception_dir, os.path.join(args.fid_checkpoint_dir, args.dataset))
        fid_score = fid.calculate_frechet_distance(val_mu, val_cov, mu, cov, eps=1e-6)
        print('| fid:', fid_score)
        print('| time:', time.time() - start_time)
    except:
        fid_score = 0

    return torch_log_reg_acc, torch_mlp_acc, torch_cnn_acc, fid_score, grid


def run_eval_pipeline_checkpoint(checkpoint_path, gen_batch_size, start, k, args):
    (val_img, val_label), metadata = fetch_data(args.dataset, args.datadir, training=False, download=False, as_array=True)
    val_img = torch.tensor(val_img) * 2 - 1.0
    val_label = torch.tensor(val_label)
    g, args = load_checkpoint_for_eval(checkpoint_path, start, k, args, metadata)
    g.eval()
    args.expdir='.'

    gen_img, gen_label = generate({'g': g}, 60000, gen_batch_size, False, args=args, metadata=metadata)
    del g
    logreg_acc, mlp_acc, cnn_acc, fid_score, sample_image = run_eval_pipeline(gen_img, gen_label, val_img, val_label, args)

    print(args.checkpoint)
    print(logreg_acc, mlp_acc, cnn_acc, fid_score)


def extract_acc(res):
    return [(acc_key, res[0][acc_key]) for acc_key in [k for k in res[0].keys() if k[:3] == 'acc']]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--datalens", default=False, action='store_true')
    parser.add_argument("--dataset", default='mnist', choices=['mnist', 'fashion_mnist', 'celeb_32_2'], type=str)
    parser.add_argument("--datadir", type=str, default='datasets')
    parser.add_argument("--inception_dir", type=str, default='inception_checkpoint')
    parser.add_argument("--fid_checkpoint_dir", type=str, default='fid_checkpoints')
    parser.add_argument("--gen_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser = add_generator_args(parser)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # set up directories
    os.makedirs(args.inception_dir, exist_ok=True)
    os.makedirs(args.fid_checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.fid_checkpoint_dir, args.dataset), exist_ok=True)

    run_eval_pipeline_checkpoint(args.checkpoint, args.gen_batch_size, 0, 0, args)













