# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from geomloss import SamplesLoss
import argparse
import sys
sys.path.append(".")
from src.rdp import calc_epsilon, sanitize_img_grads_hook, add_privacy_args, DPAdam, clip_img_grads_hook
from src.utils import experiment_init, add_experiment_args, Metric, save_checkpoint, PrivacyExceededError
from src.trainer import get_sinkhorn_eps, main_loop
from src.data import fetch_data, IIDBatchSampler, get_balanced_class_labels, IndexedDataset, add_data_args
from src.architecture import add_generator_args, build_generator, SmallDiscriminator
from src.cost_function import get_cost_adv


def compute_loss(g, d, img, label, loss_fn, args, metadata):
    loss_mets = Metric()
    batch_size = len(label)

    # concatenate label one-hot encoding to form a joint sample ~ p(x,y)
    flat_img = img.view(batch_size, -1).float()
    if args.class_cond:
        flat_img_with_y = torch.cat([flat_img,
                                     g.embed_multiplier * F.one_hot(label, metadata['label_dim']).float()], dim=1)

    # sample noise
    if args.noise == "uniform":
        z = torch.rand((batch_size, g.latent_dim)).to(img.device)
    else:
        bernoulli = torch.distributions.Bernoulli(torch.tensor([0.5]))
        z = bernoulli.sample((batch_size, g.latent_dim)).view(batch_size, g.latent_dim).to(img.device)

    if args.class_cond:
        if args.balance_class_label:
            gen_label = get_balanced_class_labels(batch_size, metadata['label_dim']).to(label.device)
        else:
            gen_label = torch.randint(metadata['label_dim'], (batch_size,)).to(label.device)
        if 'biggan' in g.tag():
            gen_y = g.shared(gen_label)
        else:
            gen_y = gen_label
    else:
        gen_label = torch.randint(metadata['label_dim'], (batch_size,)).to(label.device)
        gen_y = None

    gen_img = g(z, gen_y)


    if g.training and args.dp and args.mechanism == 'img_grad':
        gen_img.register_hook(
            lambda x: sanitize_img_grads_hook(x, args, loss_mets)
        )

    if args.debias:
        n_regen = int(args.debias_ratio*batch_size)
        n_gen = gen_img.shape[0]
        if n_regen > 0:
            alt_z = torch.rand((n_regen, g.latent_dim)).to(label.device)
            alt_gen_img = g(alt_z, gen_y[:n_regen]) # this way, the same label doesn't appear twice in calculating W(X,X)
            if g.training and args.dp and args.mechanism == 'img_grad':
                alt_gen_img.register_hook(
                    lambda x: clip_img_grads_hook(x, args, loss_mets)
                )
            gen_img = torch.cat([gen_img, alt_gen_img], dim=0)
            gen_label = torch.cat([gen_label, gen_label[:n_regen]])
    if gen_img.shape[-1] == 32 and metadata['img_dim'][-1] == 28:  # then we are using biggan on mnist/fmnist
        gen_img = torch.nn.functional.pad(gen_img, [-2, -2, -2, -2])
    if gen_img.shape[-1] == 28 and metadata['img_dim'][-1] == 32:  # then we are using convcond on celeb/cifar
        gen_img = torch.nn.functional.adaptive_avg_pool2d(gen_img, [32, 32])

    flat_gen_img = gen_img.view(gen_img.shape[0], -1)

    if args.class_cond:
        flat_gen_img_with_y = torch.cat([flat_gen_img,
                                         g.embed_multiplier * F.one_hot(gen_label, metadata['label_dim']).float()],
                                        dim=1)

    # print (gen_img.device, gen_label.device, d
    if args.d_steps > 0:
        gen_d_feat = d(gen_img, gen_label).view(gen_img.shape[0], -1)
        real_d_feat = d(img, label).view(batch_size, -1)
        gen_d_feat = (gen_d_feat / torch.norm(gen_d_feat, dim=1, keepdim=True)) * flat_gen_img.norm(
            keepdim=True).detach()
        real_d_feat = (real_d_feat / torch.norm(real_d_feat, dim=1, keepdim=True)) * flat_gen_img.norm(
            keepdim=True).detach()
        if args.class_cond:
            gen_feat = torch.cat([flat_gen_img_with_y, gen_d_feat], dim=1)
            real_feat = torch.cat([flat_img_with_y, real_d_feat], dim=1)
        else:
            gen_feat = torch.cat([flat_gen_img, gen_d_feat], dim=1)
            real_feat = torch.cat([flat_img, real_d_feat], dim=1)
    else:
        if args.class_cond:
            gen_feat = flat_gen_img_with_y
            real_feat = flat_img_with_y
        else:
            gen_feat = flat_gen_img
            real_feat = flat_img
    if args.debias:
        g_loss = 2*loss_fn(gen_feat[:n_gen], real_feat) - loss_fn(gen_feat[:n_gen], gen_feat[n_regen:])
    else:
        g_loss = loss_fn(gen_feat, real_feat)
    d_loss = -g_loss

    loss_mets.update({
        'd_loss': d_loss.data.cpu().item(),
        'g_loss': g_loss.data.cpu().item()
    })

    return g_loss, d_loss, loss_mets


def train_step(models, optimizers, img, label, idx, loss_fn, args, metadata):

    g = models['g']
    g_optim = optimizers['g_optimizer']

    # move d and d optimizer to gpu
    d = models['d']
    d_optim = optimizers['d_optimizer']
    if args.adv:
        # First, optimize D
        g.training = False
        d.training = True
        for p in d.parameters():
            p.requires_grad = True
        for p in g.parameters():
            p.requires_grad = False

        d_loss_first = 0
        d_loss_last = 0
        for i in range(args.d_steps):
            d_optim.zero_grad()
            _, d_loss, loss_mets = compute_loss(g, d, img, label, loss_fn, args, metadata)
            if i == 0:
                d_loss_first = loss_mets['d_loss']
            if i == args.d_steps - 1:
                d_loss_last = loss_mets['d_loss']
            d_loss.backward()
            d_optim.step()

        d.training = False
        for p in d.parameters():
            p.requires_grad = False

    # Take one step in G
    g.training = True
    for p in g.parameters():
        p.requires_grad = True

    g_optim.zero_grad()
    g_loss, _, loss_mets = compute_loss(g, d, img, label, loss_fn, args, metadata)
    g_loss.backward()

    if args.global_step % 20 == 0:
        eps = calc_epsilon(args.batch_size, args.noise_multiplier, args.delta, len(train_data), global_step, args.sampling)
        loss_mets.merge({'epsilon':eps})
        if eps >= args.target_epsilon:
            raise PrivacyExceededError(loss_mets, "Target privacy %f reached" % args.target_epsilon)

    if args.global_step % 100 == 0:
        grads = [p.grad.data.cpu().view(-1) for p in g.parameters() if p.requires_grad]

    g_optim.step()
    if args.adv:
        loss_mets.merge({'d_loss/first': d_loss_first, 'd_loss/last': d_loss_last})
    return loss_mets


def val_step(models, img, label, loss_fn, args, metadata):
    g = models['g']
    g.training = False
    if args.adv:
        d = models['d'].to(img.device)
    else:
        d = None
    _, _, loss_mets = compute_loss(g, d, img, label, loss_fn, args, metadata)

    return loss_mets

def get_checkpoint_writer(args, models, optimizers):
    def writer(score, epoch, global_step):
        save_checkpoint(args.expdir,
                        {'g': models['g'].state_dict(),
                         'g_optimizer': optimizers['g_optimizer'].state_dict(),
                         'd': models['d'].state_dict() if args.adv else None,
                         'd_optimizer': optimizers['d_optimizer'].state_dict() if args.adv else None,
                         },
                        score,
                        epoch, global_step,
                        args)
    return writer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # data configurations

    parser.add_argument("--num_examples", type=int, default=0, help='number of training examples, set to 0 for full '
                                                                    'training split')
    parser.add_argument("--noise", default="uniform", type=str, choices=['uniform', 'bernoulli'])
    parser.add_argument("--debias", default=False, action='store_true')
    parser.add_argument("--debias_ratio", type=float, default=1.0)
    parser.add_argument("--target_epsilon", type=float, default=0.0)
    parser.add_argument("--num_g_examples", type=int, default=10000, help='number of examples generated for validation')
    parser.add_argument("--g_batch_size", type=int, default=200, help='batch size for validation generation')
    parser.add_argument("--class_cond", default=1, type=int, help='1 to do class conditional generation, 0 for no cond')
    # training hyper params
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--optimizer", type=str, choices=['adam', 'sgd', 'rmsprop', 'adagrad'])
    parser.add_argument("--balance_class_label", action='store_true', default=False)
    parser.add_argument('--save_every_val', default=False, action='store_true')
    # adversarial params
    parser.add_argument("--adv", default=False, action="store_true")
    parser.add_argument("--d_noise_multiplier", type=float, default=0.7)
    parser.add_argument("--d_l2_norm_clip", type=float, default=0.5)
    parser.add_argument("--d_num_hidden_layers", default=1, type=int)
    parser.add_argument("--d_hidden_units", default=1000, type=int)
    parser.add_argument("--d_out_dim", default=1, type=int)
    parser.add_argument("--d_lr", default=1e-2, type=float)
    parser.add_argument("--d_steps", default=10, type=int)

    # sinkhorn loss
    parser.add_argument("--sinkhorn_eps", type=float, default=0.2)
    parser.add_argument("--p", type=float, default=2)
    parser.add_argument("--sinkhorn_eps_schedule", nargs='+', type=float, default=None)
    parser.add_argument("--sinkhorn_eps_step_every", default=None, type=int)

    # sinkhorn cost
    parser.add_argument("--cost", default="default", choices=['default', 'rkhs', 'adv_only', 'weighted', 'l1+l2'])
    parser.add_argument("--mixture_fraction", default=1.0, type=float)

    # privacy params
    parser = add_privacy_args(parser)
    parser = add_data_args(parser)

    # experiment args
    parser = add_experiment_args(parser)

    # generator and visualization arguments
    parser = add_generator_args(parser)

    args = parser.parse_args()

    print(args)

    expdir, state_dict, epoch, global_step, train_writer, val_writer, torch_rand_gen = experiment_init(args)
    args.expdir = expdir
    q = 0

    # data
    train_data, _ = fetch_data(args.dataset, args.datadir, training=True, download=True, as_array=False, num_examples=0)
    val_data, metadata = fetch_data(args.dataset, args.datadir, training=False, download=False, as_array=False, num_examples=0)
    # generator
    g = build_generator(args, metadata['img_dim'], metadata['label_dim']).to(args.device)
    assert args.sampling == 'poisson'
    num_batches_per_epoch = int(len(train_data) / args.batch_size)

    if args.adv:
        d = SmallDiscriminator(out_dim=args.d_out_dim,
                                num_hidden_layers=args.d_num_hidden_layers,
                                num_hidden_units=args.d_hidden_units,
                                in_channel=metadata['img_dim'][0]).to(args.device)
        print(d)
        print('| num params in discriminator: {}'.format(
            np.sum(
                [np.prod(p.size()) for p in
                 filter(lambda p: p.requires_grad, d.parameters())])))
    else:
        d = None

    if args.dp:
        print('| DP training is ON: ')

        assert args.mechanism == 'img_grad'
        print('| using mechanism: img_grad')
        iterations = 0
        epsilon = 0
        d_epsilon = 0

        # log dp guarantees
        if args.target_epsilon == 0:
            for i in range(args.num_epochs + 1):
                iterations = global_step - 1 + i * num_batches_per_epoch
                epsilon = calc_epsilon(args.batch_size, args.noise_multiplier, args.delta, len(train_data), iterations, args.sampling)
                val_writer.add_scalar('epsilon', epsilon, global_step=iterations)
                if args.adv:
                    d_epsilon = calc_epsilon(args.batch_size, args.d_noise_multiplier, args.delta, len(train_data), iterations, args.sampling)
                    val_writer.add_scalar('d_epsilon', d_epsilon, global_step=iterations)
        else:
            i= 0
            while True:
                iterations = global_step - 1 + i * num_batches_per_epoch
                epsilon = calc_epsilon(args.batch_size, args.noise_multiplier, args.delta, len(train_data), iterations,
                                       args.sampling)
                val_writer.add_scalar('epsilon', epsilon, global_step=iterations)
                if epsilon > args.target_epsilon:
                    break
                if i > 3000:
                    break
                i += 1
            args.num_epochs = i
            print("Target Epsilon to be reached in %d epoch" % args.num_epochs)
        if not args.adv:
            print(
            '| l2_norm_clip : {}, noise_multiplier: {}, batch_size(q): {}({}), total iterations: {}) ---> ({},{})-DP guarantee'.format(
                args.l2_norm_clip,
                args.noise_multiplier,
                args.batch_size, args.batch_size / len(train_data),
                iterations,
                epsilon,
                args.delta))
            d_optimizer = None
        else:
            print(
                '| l2_norm_clip : {}, noise_multiplier: {}/{}, batch_size(q): {}({}), total iterations: {}) ---> ({}+{},{})-DP guarantee'.format(
                    args.l2_norm_clip,
                    args.noise_multiplier,
                    args.d_noise_multiplier,
                    args.batch_size, args.batch_size / len(train_data),
                    iterations,
                    epsilon,
                    d_epsilon,
                    args.delta))
            d_optimizer = DPAdam(args.d_l2_norm_clip, args.d_noise_multiplier, params=d.parameters(),
                                 weight_decay=0.05 * args.d_lr,
                                 lr=args.d_lr)
    else:
        print('| regular optimization')
        d_optimizer = None
        if args.adv:
            d_optimizer = torch.optim.Adam(params=d.parameters(),
                                 weight_decay=0.05 * args.d_lr,
                                 lr=args.d_lr)

    train_loader = DataLoader(dataset=IndexedDataset(train_data),
                              shuffle=False,
                              num_workers=8,
                              pin_memory=True,
                              batch_sampler=IIDBatchSampler(train_data, args.batch_size, num_batches_per_epoch))

    if args.optimizer == 'sgd':
        g_optimizer = torch.optim.SGD(params=g.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.optimizer == 'rmsprop':
        g_optimizer = torch.optim.RMSprop(params=g.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.optimizer == 'adagrad':
        g_optimizer = torch.optim.Adagrad(params=g.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        g_optimizer = torch.optim.Adam(params=g.parameters(), lr=args.lr, weight_decay=1e-5)

    val_loader = DataLoader(dataset=val_data,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)

    # get sinkhorn eps
    sinkhorn_eps = get_sinkhorn_eps(epoch, args)

    # configure loss function
    if args.debias:
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", debias=False, p=args.p, blur=sinkhorn_eps, backend='tensorized',
                                    cost=get_cost_adv(args, metadata))
    else:
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", debias=True, p=args.p, blur=sinkhorn_eps, backend='tensorized',
                                cost=get_cost_adv(args, metadata))

    # load state dicts for optimizer and generator
    if state_dict is not None:
        g.load_state_dict(state_dict['g'])
        g_optimizer.load_state_dict(state_dict['g_optimizer'])
        if args.adv:
            d.load_state_dict(state_dict['d'])
            d_optimizer.load_state_dict(state_dict['d_optimizer'])

    models = {'g': g, 'd': d}
    optimizers = {'g_optimizer': g_optimizer, 'd_optimizer': d_optimizer}

    ckpt_writer = get_checkpoint_writer(args, models, optimizers)
    # run training
    main_loop(models, optimizers, train_loader, val_loader, train_step, val_step, sinkhorn_loss, train_writer,
              val_writer, ckpt_writer,
              args.dataset, epoch, global_step, args, metadata)