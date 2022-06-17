# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""
Implement loop for training and invokes visualization and evaluation steps.
"""
import numpy as np
import torch
from .pytorch_fid_utils import compute_fid, fid_preproc
from .utils import img_sample_plot, histogram_plot, Metric, save_sample_img, print_tensor_prop, PrivacyExceededError
from .data import fetch_data
from .train_mnist_classifier import torch_evaluate
import os, time, json

DEBUG=False
# function stubs to implement
def val_step(models, img, label, args, metadata):
    return None


def train_step(models, img, label, idx, args, metadata):
    return None


def generate(models, num_examples, batch_size, fixed, args, metadata):
    '''
    @param num_examples: amount to generate
    @param batch_size: generation forward pass batch size
    @param fixed: use fixed vectors or not
    '''
    assert num_examples % batch_size == 0

    g = models['g']
    device = args.device
    img = []
    label = []
    g.training=False
    bernoulli = torch.distributions.Bernoulli(torch.tensor([0.5]))
    with torch.no_grad():
        for i in range(1, int(num_examples / batch_size) + 1):
            if args.noise == "uniform":
                z_batch = torch.rand(batch_size, g.latent_dim).to(device)
            else:
                z_batch = bernoulli.sample((batch_size, g.latent_dim)).view(batch_size, g.latent_dim).to(device)
            label_batch = torch.randint(g.label_dim, (batch_size,)).to(device)

            if fixed:
                assert batch_size == num_examples and g.num_vis_examples == num_examples
                img_batch, label_batch = g.sample_labelled_fixed()  # generator handles passing to shared if needed
            else:
                img_batch, label_batch = g.sample_labelled(z_batch, label_batch)
            if img_batch.shape[-1] == 32 and metadata['img_dim'][-1] == 28:
                img_batch = torch.nn.functional.pad(img_batch, [-2, -2, -2, -2])
            if img_batch.shape[-1] == 28 and metadata['img_dim'][-1] == 32:  # then we are using convcond on celeb/cifar
                img_batch = torch.nn.functional.adaptive_avg_pool2d(img_batch, [32, 32])
            if len(img_batch.shape) == 2:
                img_batch = img_batch.view((img_batch.shape[0],) + metadata['img_dim'])
            if args.cost == 'bce':
                # then img need to pass through activation function
                img_batch = torch.sigmoid(img_batch) * 2 - 1.0
            img.append(img_batch.detach().cpu())
            label.append(label_batch.detach().cpu())

            if i % args.print_interval == 0 or i == 1:
                print('| [gen] generating examples: {}/{}'.format(i * batch_size, num_examples))

    img = torch.cat(img)
    label = torch.cat(label)
    if DEBUG:
        print_tensor_prop(img, label)
    return img, label


def vis(models, val_data_name, global_step, writer, args, metadata):

    gen_img, gen_label = generate(models,
                                     num_examples = args.num_vis_examples,
                                     batch_size = args.num_vis_examples,
                                     fixed = False,
                                     args=args,
                                    metadata=metadata)

    fixed_gen_img, fixed_gen_label = generate(models,
                                                 num_examples=args.num_vis_examples,
                                                 batch_size=args.num_vis_examples,
                                                 fixed=True,
                                                 args=args,metadata=metadata)

    # handling for PIL Image based datasets, avoid querying entire validation set.
    (val_img, val_label),_ = fetch_data(val_data_name, args.datadir, training=False, download=False, as_array=True,
                                        num_examples=args.num_vis_examples)
    # if args.dataset in ['celeb_32', 'celeb_32_2']:
    #     example_idx = torch.randint(len(val_data), (args.num_vis_examples,))
    #     val_img = torch.stack([val_data[i][0] for i in example_idx], dim=0).numpy()
    #     val_label = np.array([val_data[i][1] for i in example_idx])
    # else:
    #     val_idx = np.random.choice(len(val_data), args.num_vis_examples, replace=True)
    #     val_img, val_label = fetch_data(val_data)
    #     val_img, val_label = val_img[val_idx].numpy(), val_label[val_idx].numpy()
    if DEBUG:
        print_tensor_prop(val_img, val_label)
    met = Metric()

    if args.dataset in ['cifar', 'svhn', 'svhn_28', 'celeb_32', 'stackedmnist', 'celeb_32_2']:
        gen_img = gen_img.numpy().transpose((0, 2, 3, 1))   # swap to W x H x Channel Dim
        fixed_gen_img = fixed_gen_img.numpy().transpose((0, 2, 3, 1))
        val_img = val_img.transpose((0, 2, 3, 1))
        gen_img, gen_label = gen_img, gen_label.numpy()
        fixed_gen_img, fixed_gen_label = fixed_gen_img, fixed_gen_label.numpy()
    else:
        fixed_gen_img, fixed_gen_label = fixed_gen_img.numpy(), fixed_gen_label.numpy()
        gen_img, gen_label = gen_img.numpy(), gen_label.numpy()
        gen_img = gen_img[:, 0, :, :]  # remove channel dim
        fixed_gen_img = fixed_gen_img[:, 0, :, :]
        val_img = val_img[:, 0, :, :]

    fixed_gen_img = fixed_gen_img * 0.5 + 0.5
    gen_img = gen_img * 0.5 + 0.5


    met['gen_img_examples'] = (None,
                               img_sample_plot(gen_img, gen_label, metadata['cmap']),
                               'img')
    met['fixed_gen_img_examples'] = (None,
                                     img_sample_plot(fixed_gen_img, fixed_gen_label, metadata['cmap']),
                                     'img')
    met['val_img_examples'] = (None,
                               img_sample_plot(val_img, val_label, metadata['cmap']),
                               'img')

    met.publish(writer, global_step-1)


def eval(models, val_data_name, global_step, writer, args, metadata):
    gen_img, gen_label = generate(models,
                                 num_examples=args.num_g_examples,
                                 batch_size=args.g_batch_size,
                                 fixed=False,
                                 args=args, metadata=metadata)
    print('| generated {} synthetic examples'.format(len(gen_label)))
    met = Metric()
    gen_img = gen_img.to(args.device)
    gen_label = gen_label.to(args.device)
    # fid score
    fid = compute_fid(fid_preproc(gen_img, (0.5, 0.5)), args)
    met['fid'] = fid
    if args.class_cond:

        (val_img, val_label), _ = fetch_data(val_data_name, args.datadir, training=False, download=False, as_array=True)
        val_img = torch.tensor(val_img) * 2 - 1.0
        val_label = torch.tensor(val_label)
        print('| {} examples for classifier testing'.format(len(val_label)))
        val_img = val_img.to(args.device)
        val_label = val_label.to(args.device)
        # label distribution histograms
        all_accs = []
        for i in range(1):
            if DEBUG:
                print_tensor_prop(val_img, val_label)
            mlp_acc, _ = torch_evaluate(gen_img, gen_label, val_img, val_label, args, 'MLP', patience=10)
            cnn_acc, _ = torch_evaluate(gen_img, gen_label, val_img, val_label, args, 'CNN', patience=10)
            log_acc, _ = torch_evaluate(gen_img, gen_label, val_img, val_label, args, 'LogReg', patience=10)
            all_accs.append([mlp_acc, cnn_acc, log_acc])

        all_accs = np.array(all_accs)
        met['mlp_acc_torch'] = all_accs.mean(axis=0)[0]
        met['log_reg_acc_torch'] =  all_accs.mean(axis=0)[1]
        met['cnn_acc_torch'] =  all_accs.mean(axis=0)[2]

        gen_label_dist = np.bincount(gen_label.cpu(), minlength=metadata['label_dim']) / len(gen_label)
        met['gen_label_dist'] = (gen_label_dist,
                                 histogram_plot(gen_label_dist, 'generated data label distribution'),
                                 metadata['display_labels'],
                                 'hist')

        val_label_dist = np.bincount(val_label.cpu(), minlength=metadata['label_dim']) / len(val_label)
        met['val_label_dist'] = (val_label_dist,
                                 histogram_plot(val_label_dist, 'val data label distribution'),
                                 metadata['display_labels'],
                                 'hist')

    met.print_metric()
    met.publish(writer, global_step - 1)

    return met


def val(models, loader, val_step_fn, loss_fn, epoch, global_step, writer, args, metadata):
    num_batches = len(loader)
    met = Metric()
    for i, batch in enumerate(loader, start=1):
        batch = tuple([x.to(args.device) for x in batch])
        batch_met = val_step_fn(models, *batch, loss_fn, args, metadata)
        if i % args.print_interval == 0 or i == 1:
            print('| [val] epoch {}, step {}/{}:'.format(epoch-1, i, num_batches), batch_met.make_string())
        met = met.accumulate(batch_met)

    met = met.div(num_batches)
    met.publish(writer, global_step - 1)
    met.print_metric()


def train(models, optimizers, loader, train_step_fn, loss_fn, epoch, global_step, writer, args, metadata):
    num_batches = len(loader)
    mean_loss = 0
    for i, batch, in enumerate(loader, start=1):
        batch_size = len(batch[0])

        batch = tuple([x.to(args.device) for x in batch])
        args.global_step = global_step
        try:
            met = train_step_fn(models, optimizers, *batch, loss_fn, args, metadata)
            met.merge({'epoch': epoch,
                                 'batch_size': batch_size
                                })
            mean_loss += met['g_loss']
        except PrivacyExceededError as err:
            print(err.message)
            met = err.met
            met.merge({'epoch': epoch,
                       'batch_size': batch_size
                       })
            mean_loss += met['g_loss']
            met.publish(writer, global_step)
            print('| epoch {}, step {}/{}, global step {}:'.format(epoch, i, num_batches, global_step),
                  met.make_string())
            global_step += 1
            raise PrivacyExceededError(None, err.message)
        if i % args.print_interval == 0 or i == 1:
            args.train_losses.append(met['g_loss'])
            met.publish(writer, global_step)
            print('| epoch {}, step {}/{}, global step {}:'.format(epoch, i, num_batches, global_step),
                  met.make_string())
        global_step += 1
    args.train_losses_mean.append(mean_loss/num_batches)
    return global_step

def get_sinkhorn_eps(epoch, args):
    if args.sinkhorn_eps_schedule is None:
        return args.sinkhorn_eps
    assert args.sinkhorn_eps_step_every is not None
    sinkhorn_eps_idx = min((epoch-1) // args.sinkhorn_eps_step_every, len(args.sinkhorn_eps_schedule)-1)
    return args.sinkhorn_eps_schedule[sinkhorn_eps_idx]


def main_loop(models, optimizers, train_loader, val_loader, train_step, val_step, loss_fn, train_writer, val_writer,
              checkpoint_writer, val_data_name, start_epoch, global_step, args, metadata):
    best_score = 0
    start_time = time.time()
    epoch = start_epoch
    g = models['g']
    if DEBUG:
        g.eval()
        with torch.no_grad():
            val(models, val_loader, val_step, loss_fn, epoch, global_step, val_writer, args, metadata)
        met = eval(models, val_data_name, global_step, val_writer, args, metadata)
        score = met['fid']
        print('| epoch {} validate time: {}'.format(epoch, time.time() - start_time))

    args.train_losses = []
    args.train_losses_mean = []
    for e in range(epoch, epoch + args.num_epochs):
        start_time = time.time()
        print('| start training epoch {}, global step {}'.format(e, global_step))

        # sinkhorn_eps schedule
        sinkhorn_eps = get_sinkhorn_eps(e, args)
        loss_fn.blur = sinkhorn_eps

        g.train()
        try:
            global_step = train(models, optimizers, train_loader, train_step, loss_fn, e, global_step,
                                train_writer, args, metadata)
            print('| epoch {} train time: {}'.format(e, time.time() - start_time))
        except PrivacyExceededError as err:
            print('| epoch {} train time: {}'.format(e, time.time() - start_time))
            print("Privacy budget reached, finishing up")
            with torch.no_grad():
                vis(models, val_data_name, global_step, val_writer, args, metadata)
            print('| epoch {} vis time: {}'.format(e, time.time() - start_time))

            start_time = time.time()
            print('| validating for epoch {}, global step {}'.format(e, global_step - 1))
            g.eval()
            with torch.no_grad():
                val(models, val_loader, val_step, loss_fn, e, global_step, val_writer, args, metadata)
                met = eval(models, val_data_name, global_step, val_writer, args, metadata)
                #score = met['fid']
                score = (met['mlp_acc_torch'] + met['log_reg_acc_torch'] + met['cnn_acc_torch']) * 100 / 3 - met['fid']
            print('| epoch {} validate time: {}'.format(e, time.time() - start_time))

            if score > best_score:
                best_score = score
                if not args.save_every_val:
                    checkpoint_writer(1, 'best', 1)
            break

        # visualization
        if e % args.vis_interval == 0 or e == 1:
            with torch.no_grad():
                vis(models, val_data_name, global_step, val_writer, args, metadata)
            print('| epoch {} vis time: {}'.format(e, time.time() - start_time))

        # evaluation
        if e % args.val_interval == 0 or e == epoch + args.num_epochs - 1:
            start_time = time.time()
            print('| validating for epoch {}, global step {}'.format(e, global_step - 1))
            g.eval()
            with torch.no_grad():
                val(models, val_loader, val_step, loss_fn, e, global_step, val_writer, args, metadata)
                met = eval(models, val_data_name, global_step, val_writer, args, metadata)
                score = (met['mlp_acc_torch']+met['log_reg_acc_torch']+met['cnn_acc_torch']) * 100 / 3 - met['fid']
            #score = met['fid']
            print('| epoch {} validate time: {}'.format(e, time.time() - start_time))
            save_sample_img(models, metadata['label_dim'], 10, args, global_step)
            if args.save_every_val:
                checkpoint_writer(score, e, global_step)

            if score > best_score:
                best_score = score
                if not args.save_every_val:
                    checkpoint_writer(1, 'best', 1)

    checkpoint_writer(1, 'last', 1)
    save_sample_img(models, metadata['label_dim'], 10, args, -1)
    np.save(os.path.join(args.expdir, 'losses.npy'), np.array(args.train_losses))
    np.save(os.path.join(args.expdir, 'losses_mean_epoch.npy'), np.array(args.train_losses_mean))
    with open(os.path.join(args.expdir, "results.json"), "w") as f:
        json.dump({'best_score': best_score}, f)