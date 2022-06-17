# Pytorch Implementation of "Don't Generate Me: Training Differentially Private Generative Models with Sinkhorn Divergence" (Neurips 2021)

Tianshi Cao, Alex Bie, Arash Vahdat, Sanja Fidler, Karsten Kreis

[Paper](https://arxiv.org/abs/2111.01177), [Project Page](https://nv-tlabs.github.io/DP-Sinkhorn/)

## Environment Setup

This code has been tested on [TensorFlow container](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_20-08.html) with tag `20.08-tf1-py3`. 
If you don't have Docker and Nvidia Container Toolkit installed, follow one of the many installation guides online (e.g. [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)) to install it.
Then, follow instructions [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) to download and run the container (make sure to pull the correct tag by browsing the <b>Tags</b> tab). 
From inside the container, install additional packages (i.e. pytorch) through `pip install -r requirements.txt`.

Alternatively, you may build TensorFlow 1.15 natively on your platform, and install packages in requirements.txt as usual.

## Commands for training DP-Sinkhorn
From the root directory of the repository, run `sinkhorn_poisson.py` to train DP-Sinkhorn. 
On first execution, it will download datasets to path specified by `--datadir`, which defaults to `{REPOSITORY_ROOT}/datasets`. 
Logs and checkpoints are saved to the path specified by `--resultdir`. By default this is `{REPOSITORY_ROOT}/results`. 
A new log folder will be created for each experiment with a different `exp_name` in `{REPOSITORY_ROOT}/results/exp_name`.
The last checkpoint will be save as `last_1_1.00000.pt` in the log folder.
Information such as training loss curves and visualization of generated images are logged to tensorboard in that folder as well.

For training DP-Sinkhorn (m=1) on MNIST: 
```
python sinkhorn_poisson.py mnist_debias_2_l1l2_m1 --dataset mnist --resultdir result/mnist_dp --target_epsilon 10.0 --debias --debias_ratio 0.2 --cost l1+l2 --p 1 --mixture_fraction 1.0 --val_interval 100 --vis_interval 40  --num_epochs 400 --batch_size 50 --lr 1e-5 --sinkhorn_eps 0.05 --g_type conv_cond --act_function lllt --latent_dim 16 --hidden_dim 64 --embed_dim 8 --embed_multiplier 15 --sampling poisson --mechanism img_grad --datadir datasets --num_vis_examples 20 --d_steps 0 --dp --noise_multiplier 1.1 --l2_norm_clip 0.5 --sensitivity_to_l2_norm_clip 2
```
For training DP-Sinkhorn (m=3) on MNIST:
```
python sinkhorn_poisson.py mnist_debias_2_l1l2_m3 --dataset mnist --resultdir result/mnist_dp --target_epsilon 10.0 --debias --debias_ratio 0.2 --cost l1+l2 --p 1 --mixture_fraction 3.0 --val_interval 100 --vis_interval 40  --num_epochs 400 --batch_size 50 --lr 1e-5 --sinkhorn_eps 0.05 --g_type conv_cond --act_function lllt --latent_dim 16 --hidden_dim 64 --embed_dim 8 --embed_multiplier 15 --sampling poisson --mechanism img_grad --datadir datasets --num_vis_examples 20 --d_steps 0 --dp --noise_multiplier 1.1 --l2_norm_clip 0.5 --sensitivity_to_l2_norm_clip 2
```
For training DP-Sinkhorn (m=1) on FashionMNIST:
```
python sinkhorn_poisson.py fashion_mnist_debias_2_l1l2_m1 --dataset fashion_mnist --resultdir result/fmnist_dp --target_epsilon 10.0 --debias --debias_ratio 0.2 --cost l1+l2 --p 1 --mixture_fraction 1.0 --val_interval 100 --vis_interval 40  --num_epochs 400 --batch_size 50 --lr 1e-5 --sinkhorn_eps 0.05 --g_type conv_cond --act_function lllt --latent_dim 16 --hidden_dim 64 --embed_dim 8 --embed_multiplier 15 --sampling poisson --mechanism img_grad --datadir datasets --num_vis_examples 20 --d_steps 0 --dp --noise_multiplier 0.9 --l2_norm_clip 0.5 --sensitivity_to_l2_norm_clip 2
```
For training DP-Sinkhorn (m=3) on FashionMNIST:
```
python sinkhorn_poisson.py fashion_mnist_debias_2_l1l2_m3 --dataset fashion_mnist --resultdir result/fmnist_dp --target_epsilon 10.0 --debias --debias_ratio 0.2 --cost l1+l2 --p 1 --mixture_fraction 3.0 --val_interval 100 --vis_interval 40  --num_epochs 400 --batch_size 50 --lr 1e-5 --sinkhorn_eps 0.05 --g_type conv_cond --act_function lllt --latent_dim 16 --hidden_dim 64 --embed_dim 8 --embed_multiplier 15 --sampling poisson --mechanism img_grad --datadir datasets --num_vis_examples 20 --d_steps 0 --dp --noise_multiplier 0.9 --l2_norm_clip 0.5 --sensitivity_to_l2_norm_clip 2
```
For training DP-Sinkhorn on CelebA:
```
python sinkhorn_poisson.py celeb_cond_b200_e5 --dataset celeb_32_2 --resultdir result/celeb_32_2 --val_interval 40 --vis_interval 40 --num_epochs 260 --target_epsilon 10.0 --batch_size 200 --lr 1e-5 --sinkhorn_eps 5 --cost l2 --p 2 --g_type cond_biggan --act_function lllt --latent_dim 32 --hidden_dim 64 --embed_dim 4 --embed_multiplier 15 --sampling poisson --mechanism img_grad --datadir datasets --noise_multiplier 0.8 --l2_norm_clip 0.1 --sensitivity_to_l2_norm_clip 2 --dp
```
### Evaluation
Use `finalize_result.py` to evaluate DP-Sinkhorn generators for downstream classifier accuracy and FID. 
Set the first argument to the path of the generator checkpoint, and use the same generator configuration as used in training. 
For example, for evaluating DP-Sinkhorn (m=1) on MNIST, the checkpoint path is `mnist_dp/mnist_debias_2_l1l2_m1/last_1_1.00000.pt`,
use the command:
```
python finalize_result.py results/mnist_dp/mnist_debias_2_l1l2_m1/last_1_1.00000.pt" --dataset mnist --g_type conv_cond --act_function lllt --latent_dim 16 --hidden_dim 64 --embed_dim 8 --embed_multiplier 15 --datadir datasets --gen_batch_size 500 --seed 0
```
For FID computation, we provide precomputed features of the real datasets in the `fid_checkpoints` folder. If these files are not present, `finalize_result.py` will iterate through the test set and recompute the features.
### Citation
If you found this codebase useful in your research, please consider citing
```
@inproceedings{Cao2021DPSinkhorn,
title = {Don't Generate Me: Training Differentially Private Generative Models with Sinkhorn Divergence}, 
author = {Tianshi Cao and Alex Bie and Arash Vahdat and Sanja Fidler and Karsten Kreis},
booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
year = {2021}
}
```
