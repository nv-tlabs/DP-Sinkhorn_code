# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
from .biggan.biggan import BigGan, Discriminator, SmallDiscriminator
from .dcgan import ConvCondGenerator, ConvCondSqGenerator

def add_generator_args(parser):
    parser.add_argument("--g_type", default='joint',
                        choices=[ConvCondGenerator.tag(),
                                 ConvCondSqGenerator.tag(),
                                 BigGan.tag()])
    parser.add_argument("--embed_dim", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--embed_multiplier", type=float, default=15)
    parser.add_argument("--act_function", default="tttn")
    return parser


def build_generator(args, img_dim, label_dim):
    if args.g_type == ConvCondGenerator.tag():
        g = ConvCondGenerator(
                args.latent_dim, img_dim, label_dim, args.embed_multiplier, args.num_vis_examples,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                fs=args.act_function,
                class_cond=args.class_cond
            )
    elif args.g_type == 'cond_biggan':
        g = BigGan(G_ch=args.hidden_dim, latent_dim=args.latent_dim, resolution=img_dim[-1],
                   G_shared=False, no_optim=True, num_vis_examples=args.num_vis_examples,
                   label_dim=label_dim,
                   embed_multiplier=args.embed_multiplier, n_channels=img_dim[0],
                   class_cond=args.class_cond)

    elif args.g_type == ConvCondSqGenerator.tag():
        g = ConvCondSqGenerator(
                args.latent_dim, img_dim, label_dim, args.embed_multiplier, args.num_vis_examples,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                fs=args.act_function
            )

    print('| using {} generator'.format(g.name()))
    print(g)
    print('| number of trainable parameters: {}'.format(g.num_trainable_params))
    return g
