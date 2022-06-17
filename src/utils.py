# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""
Implements importable logging, persistence, and visualization tools
"""
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import io
import torchvision
import numpy as np
import os
from datetime import datetime
from argparse import Namespace
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from matplotlib.figure import Figure
from torchvision.utils import save_image

def add_experiment_args(parser):

    # experiment configs
    parser.add_argument("exp_name", type=str, default='exp')
    parser.add_argument("--device", type=str, default='cuda', choices=['cpu', 'cuda', ] +
                                                                      ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resultdir", type = str, default = 'results')
    parser.add_argument("--override_expdir", type=str, default = None) # for hyperparameter search
    parser.add_argument("--checkpoint_path", type=str, default = None)
    parser.add_argument("--num_epochs", type = int, default = 500)

    parser.add_argument("--val_interval", type=int, default = 50)
    parser.add_argument("--print_interval", type=int, default=20)
    parser.add_argument("--vis_interval", type=int, default=10, help="run visualization every interval epochs")
    parser.add_argument("--num_vis_examples", type=int, default=20)
    return parser

def save_checkpoint(path, state_dict, score, epoch, global_step, args):
    torch.save({
        'args': vars(args),
        'state_dict': state_dict,
        'epoch': epoch,
        'global_step': global_step
    }, os.path.join(path, "{}_{}_{:.5f}.pt".format(epoch, global_step, score)))


def load_checkpoint(path, args):
    checkpoint = torch.load(path)
    args_dict = checkpoint['args']

    for k in args_dict:
        if k in args.__dict__ and args_dict[k] != args.__dict__[k] and k!='checkpoint_path':
            print ('| WARNING: \'{}\' in args has value: \'{}\', checkpoint value: \'{}\''.format(k, args.__dict__[k], args_dict[k]))

    return checkpoint['epoch'], checkpoint['global_step'], checkpoint['state_dict']


def load_checkpoint_with_args(path):
    checkpoint = torch.load(path)

    args = Namespace()
    args.__dict__.update(checkpoint['args'])
    return args, checkpoint['epoch'], checkpoint['global_step'], checkpoint['state_dict']


def get_expdir(exp_name, resultdir = 'workspace/runs/mnist'):
    if os.name == "posix":
        return os.path.join(resultdir, exp_name)
    else:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return os.path.join(resultdir, exp_name + "_" + current_time)# + "_" + socket.gethostname())


def experiment_init(args):

    # checkpoint loading
    epoch = 1
    global_step = 1
    state_dict = None
    if args.checkpoint_path is not None:
        print('| loading checkpoint at {}'.format(args.checkpoint_path))
        epoch, global_step, state_dict = load_checkpoint(args.checkpoint_path, args)
        print('| restart training at epoch {}, global step {}'.format(epoch, global_step))

    # logging
    expdir = args.override_expdir if args.override_expdir is not None else get_expdir(args.exp_name, args.resultdir)
    os.makedirs(expdir, exist_ok=True)

    # write original arguments to text file
    arg_strs = []
    with open(os.path.join(expdir, "args.txt"), "w") as fp:
        for arg in vars(args):
            arg_str = "%s: %s" % (arg, str(getattr(args, arg)))
            fp.write(arg_str+'\n')
            arg_strs.append(arg_str)

    train_writer = SummaryWriter(expdir+'/train')
    val_writer = SummaryWriter(expdir+'/val')

    # log args to tensorboard
    val_writer.add_text('args', ', '.join(arg_strs), global_step=global_step)

    # set seed
    np.random.seed(args.seed)
    torch_rand_gen = torch.random.manual_seed(args.seed)

    return expdir, state_dict, epoch, global_step, train_writer, val_writer, torch_rand_gen


def histogram_plot(values, title, display_labels=None):
    values = np.round(values, 3)
    xlocs = np.arange(0, len(values))
    if display_labels is None:
        display_labels = [str(x) for x in xlocs]
    fig = plt.figure(figsize=(8, 5))
    plt.bar(display_labels, values)
    for i, v in enumerate(values):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    t = torchvision.transforms.ToTensor()
    im2 = t(im)
    buf.close()
    return im2



def confusion_matrix_plot(values, display_labels=None):
    values = np.round(values, 3)
    plt.rcParams["figure.figsize"] = [12, 9]
    disp = ConfusionMatrixDisplay(values, display_labels=display_labels)
    disp.plot()
    fig = disp.figure_
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    t = torchvision.transforms.ToTensor()
    im2 = t(im)
    buf.close()
    return im2


def img_sample_plot(img, label, cmap):

    num_samples = len(img)
    assert num_samples % 5 == 0
    fig, ax = plt.subplots(int(num_samples/5), 5)
    fig.tight_layout(pad=0)
    fig.subplots_adjust(hspace=0.2, wspace=0.05)

    for i in range(int(num_samples/5)):
        for j in range(5):
            ax[i, j].imshow(img[i*5+j], cmap)
            #if display_labels is None:
            ax[i, j].set_xlabel(label[i*5+j].item())

            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)

    t = torchvision.transforms.ToTensor()
    im2 = t(im)
    buf.close()
    return im2


class Metric(dict):
    blacklist = ['epoch', 'global_step']

    def merge(self, target):
        self.update(target)
        return self

    def accumulate(self, acc):
        if acc == {}:
            return self
        else:
            items = []
            for k in acc.keys():
                if k in self.keys():
                    item = (k, self[k] + acc[k])
                else:
                    item = (k, acc[k])
                items.append(item)
            self.update(items)
            return self

    def div(self, s):
        self.update([(k, v / s) for k, v in self.items()])
        return self

    def publish(self, writer, global_step):
        for k,v in self.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(k, v, global_step=global_step)
            elif isinstance(v[1], Figure):
                writer.add_figure(k, v[1], global_step=global_step)
            elif isinstance(v[1], torch.Tensor):
                writer.add_image(k, v[1], global_step=global_step)
            elif isinstance(v[1], Image.Image):
                writer.add_image(k, v[1], global_step=global_step)

    def print_metric(self):
        for k,v in self.items():
            if isinstance(v, (int, float)):
                print('| {}:'.format(k), v)
            elif v[2] == 'hist':
                print('| {}:'.format(k))
                print(np.vstack([np.arange(0, len(v[0])), np.round(v[0],3)]).transpose())

    def make_string(self):
        strings = ['{} {}'.format(x[0],np.round(x[1],3)) for x in self.items()
                   if isinstance(x[1], (int, float)) and x[0] not in Metric.blacklist]
        return ', '.join(strings)


def save_sample_img(models, num_labels, num_samples, args, global_step):
    g = models['g']
    g.training=False
    img = []
    label = []
    for l in range(num_labels):
        z_batch = torch.rand(num_samples, g.latent_dim).cuda()
        label_batch = l * torch.ones(num_samples).long().cuda()
        print(z_batch.shape)
        img_batch, label_batch = g.sample_labelled(z_batch, label_batch)
        if args.cost == 'bce':
            # then img need to pass through activation function
            img_batch = torch.sigmoid(img_batch) * 2 - 1.0
        if img_batch.shape[-1] == 32:
            img_batch = torch.nn.functional.pad(img_batch, [-2, -2, -2, -2])
        elif img_batch.shape[-1] == 30:
            img_batch = torch.nn.functional.pad(img_batch, [-1, -1, -1, -1])
        img.append(img_batch.detach().cpu())
        label.append(label_batch.detach().cpu())
    img = torch.stack(img)
    img = torch.transpose(img, 0, 1)
    img = img.reshape(img.shape[0]*img.shape[1], img.shape[2], img.shape[3], img.shape[4])
    label = torch.cat(label)

    save_image(img, args.expdir + '/vis_img_%d.png' % global_step, nrow=num_samples, range=(-1,1), normalize=True)


def print_tensor_prop(*inputs):
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor):
            dtype = inp.dtype
            device = inp.device
            minmax = (inp.max().cpu().item(), inp.min().cpu().item())
            print("Arg %i, type %s Tensor, device %s, range %.2f to %.2f, shape %s" % (i, dtype, device, minmax[0], minmax[1], str([*inp.shape])))
        elif isinstance(inp, np.ndarray):
            print("Arg %i, type %s array, range %.2f to %.2f, shape %s" % (i, inp, np.max(inp), np.min(inp), str(inp.shape)))


class PrivacyExceededError(Exception):
    def __init__(self, met, message):
        self.met = met
        self.message = message
        super(PrivacyExceededError).__init__()