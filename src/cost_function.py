# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import numpy as np

def RBF_Kernel(fx, fy, gamma):
    "Returns the matrix of $exp(-gamma * |x_i-y_j|^2)$."
    if fx.dim() == 2:
        x_col = fx.flatten(1, -1).unsqueeze(1)
        y_lin = fy.flatten(1, -1).unsqueeze(0)
        c2 = torch.norm(x_col - y_lin, p=2, dim=2)
    else:
        x_col = fx.flatten(2, -1).unsqueeze(2)  # B N (W, H) or D -> B N 1 D
        y_lin = fy.flatten(2, -1).unsqueeze(1)  # B M (W, H) or D -> B 1 M D
        c2 = torch.norm(x_col - y_lin, p=2, dim=3)  # B N M
    # c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    RBF_K = torch.exp(-gamma * c2)  # B N M or N M
    return RBF_K

def RKHS_Norm(x, y, gamma=0.5):
    Kxy = RBF_Kernel(x, y, gamma)
    return 1 + 1 - 2 * Kxy


def get_cost_adv(args, metadata):
    if args.cost == "adv_only":
        def cost(x, y):
            if x.dim() == 2:
                x = x[:, np.prod(metadata['img_dim']):]
                y = y[:, np.prod(metadata['img_dim']):]
                c_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
                c_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
                c_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
            elif x.dim() == 3:  # Batch computation
                x = x[:, :, np.prod(metadata['img_dim']):]
                y = y[:, :, np.prod(metadata['img_dim']):]
                c_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
                c_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
                c_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)
            else:
                print("x.shape : ", x.shape)
                raise ValueError("Incorrect number of dimensions")
            return c_xx - 2 * c_xy + c_yy
    elif args.cost == 'rkhs':
        def cost(x, y):
            if x.dim() == 2:
                xi = x[:, :np.prod(metadata['img_dim']) + metadata['label_dim']]
                yi = y[:, :np.prod(metadata['img_dim']) + metadata['label_dim']]
                c_xx = (xi * xi).sum(-1).unsqueeze(1)  # (N,1)
                c_xy = torch.matmul(xi, yi.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
                c_yy = (yi * yi).sum(-1).unsqueeze(0)  # (1,M)

                xj = x[:, np.prod(metadata['img_dim']) + metadata['label_dim']:]
                yj = y[:, np.prod(metadata['img_dim']) + metadata['label_dim']:]
                r_xx = RKHS_Norm(xj, xj)
                r_xy = RKHS_Norm(xj, yj)
                r_yy = RKHS_Norm(yj, yj)

            elif x.dim() == 3:  # Batch computation
                xi = x[:, :, :np.prod(metadata['img_dim']) + metadata['label_dim']]
                yi = y[:, :, :np.prod(metadata['img_dim']) + metadata['label_dim']]
                c_xx = (xi * xi).sum(-1).unsqueeze(2)  # (B,N,1)
                c_xy = torch.matmul(xi, yi.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
                c_yy = (yi * yi).sum(-1).unsqueeze(1)  # (B,1,M)
                xj = x[:, :, np.prod(metadata['img_dim']) + metadata['label_dim']:]
                yj = y[:, :, np.prod(metadata['img_dim']) + metadata['label_dim']:]
                r_xx = RKHS_Norm(xj, xj)
                r_xy = RKHS_Norm(xj, yj)
                r_yy = RKHS_Norm(yj, yj)
            else:
                print("x.shape : ", x.shape)
                raise ValueError("Incorrect number of dimensions")
            return args.mixture_fraction * (c_xx - 2 * c_xy + c_yy) + (r_xx - 2 * r_xy + r_yy)
    elif args.cost == 'weighted':
        def cost(x, y):
            if x.dim() == 2:
                xi = x[:, :np.prod(metadata['img_dim']) + metadata['label_dim']]
                yi = y[:, :np.prod(metadata['img_dim']) + metadata['label_dim']]
                c_xx = (xi * xi).sum(-1).unsqueeze(1)  # (N,1)
                c_xy = torch.matmul(xi, yi.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
                c_yy = (yi * yi).sum(-1).unsqueeze(0)  # (1,M)

                xj = x[:, np.prod(metadata['img_dim']) + metadata['label_dim']:]
                yj = y[:, np.prod(metadata['img_dim']) + metadata['label_dim']:]
                a_xx = (xj * xj).sum(-1).unsqueeze(1)  # (N,1)
                a_xy = torch.matmul(xj, yj.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
                a_yy = (yj * yj).sum(-1).unsqueeze(0)  # (1,M)

            elif x.dim() == 3:  # Batch computation
                xi = x[:, :, :np.prod(metadata['img_dim']) + metadata['label_dim']]
                yi = y[:, :, :np.prod(metadata['img_dim']) + metadata['label_dim']]
                c_xx = (xi * xi).sum(-1).unsqueeze(2)  # (B,N,1)
                c_xy = torch.matmul(xi, yi.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
                c_yy = (yi * yi).sum(-1).unsqueeze(1)  # (B,1,M)
                xj = x[:, :, np.prod(metadata['img_dim']) + metadata['label_dim']:]
                yj = y[:, :, np.prod(metadata['img_dim']) + metadata['label_dim']:]
                a_xx = (xj * xj).sum(-1).unsqueeze(2)  # (B,N,1)
                a_xy = torch.matmul(xj, yj.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
                a_yy = (yj * yj).sum(-1).unsqueeze(1)  # (B,1,M)
            else:
                print("x.shape : ", x.shape)
                raise ValueError("Incorrect number of dimensions")
            return  (c_xx - 2 * c_xy + c_yy) + args.mixture_fraction * (a_xx - 2 * a_xy + a_yy)
    elif args.cost =='l1+l2':
        def cost(x, y):
            if x.dim() == 2:
                n = x.shape[0]
                m = y.shape[0]
                d = x.shape[1]

                x = x.unsqueeze(1).expand(n, m, d)
                y = y.unsqueeze(0).expand(n, m, d)
                dist = torch.norm(x - y, p=2, dim=2)**2 + args.mixture_fraction * torch.norm(x - y, p=1, dim=2)
            elif x.dim() == 3:
                num_batch = x.shape[0]
                n = x.shape[1]
                m = y.shape[1]
                d = x.shape[2]

                x = x.unsqueeze(2).expand(num_batch, n, m, d)
                y = y.unsqueeze(1).expand(num_batch, n, m, d)
                dist = torch.norm(x - y, p=2, dim=3)**2 + args.mixture_fraction * torch.norm(x - y, p=1, dim=3)
            else:
                print("x.shape : ", x.shape)
                raise ValueError("Incorrect number of dimensions")
            return dist
    elif args.cost =='bce':
        import torch.nn as nn
        class BCEcost(nn.Module):
            def __init__(self):
                super(BCEcost, self).__init__()
                self.loss = nn.BCEWithLogitsLoss(reduction='none')
                self.split = np.prod(metadata['img_dim'])
            def forward(self, x, y):

                if x.dim() == 2:
                    n = x.shape[0]
                    m = y.shape[0]
                    d = x.shape[1]

                    x = x.unsqueeze(1).expand(n, m, d)
                    y = y.unsqueeze(0).expand(n, m, d)

                    xi = x[:, :, :self.split]
                    yi = y[:, :, :self.split]
                    xj = x[:, :, self.split:]
                    yj = y[:, :, self.split:]
                    if args.dynamic_binarize:
                        if yi.max() > 1.0 or yi.min < 0:
                            yi = torch.bernoulli(torch.sigmoid(yi))
                        else:
                            yi = torch.bernoulli(yi)

                    dist = self.loss(xi, yi).sum(-1) + torch.norm(xj-yj, p=2, dim=2)**2

                elif x.dim() == 3:
                    num_batch = x.shape[0]
                    n = x.shape[1]
                    m = y.shape[1]
                    d = x.shape[2]

                    x = x.unsqueeze(2).expand(num_batch, n, m, d)
                    y = y.unsqueeze(1).expand(num_batch, n, m, d)

                    xi = x[:, :, :, :self.split]
                    yi = y[:, :, :, :self.split]
                    xj = x[:, :, :, self.split:]
                    yj = y[:, :, :, self.split:]
                    if args.dynamic_binarize:
                        if yi.max() > 1.0 or yi.min() < 0.0:
                            yi = torch.bernoulli(torch.sigmoid(yi))
                        else:
                            yi = torch.bernoulli(yi)
                    yi = torch.clamp(yi, 0.0, 1.0)
                    dist = self.loss(xi, yi).sum(-1) + torch.norm(xj - yj, p=2, dim=3)**2
                else:
                    print("x.shape : ", x.shape)
                    raise ValueError("Incorrect number of dimensions")
                return dist
        cost = BCEcost()
    elif args.cost =='cebce':
        #TODO: fix issue with inf label loss
        import torch.nn as nn
        class CEBCEcost(nn.Module):
            def __init__(self):
                super(CEBCEcost, self).__init__()
                self.loss = nn.BCEWithLogitsLoss(reduction='none')
                self.loss2 = nn.CrossEntropyLoss(reduction='none')
                self.split = np.prod(metadata['img_dim'])
            def forward(self, x, y):
                if x.dim() == 2:
                    n = x.shape[0]
                    m = y.shape[0]
                    d = x.shape[1]

                    x = x.unsqueeze(1).expand(n, m, d)
                    y = y.unsqueeze(0).expand(n, m, d)

                    xi = x[:, :, :self.split]
                    yi = y[:, :, :self.split]
                    xj = x[:, :, self.split:]
                    yj = y[:, :, self.split:]


                    dist = self.loss(xi, yi).sum(-1) + torch.norm(xj-yj, p=2, dim=2)**2

                elif x.dim() == 3:
                    num_batch = x.shape[0]
                    n = x.shape[1]
                    m = y.shape[1]
                    d = x.shape[2]

                    x = x.unsqueeze(2).expand(num_batch, n, m, d)
                    y = y.unsqueeze(1).expand(num_batch, n, m, d)

                    xi = x[:, :, :, :self.split]
                    yi = y[:, :, :, :self.split]
                    xj = x[:, :, :, self.split:]
                    yj = y[:, :, :, self.split:]

                    dist = self.loss(xi, yi).sum(-1) + torch.norm(xj - yj, p=2, dim=3)**2
                else:
                    print("x.shape : ", x.shape)
                    raise ValueError("Incorrect number of dimensions")
                return dist
        cost = CEBCEcost()
    else:
        print("Using Default Cost Elementary Function")
        if args.p == 2:
            def cost(x, y):
                if x.dim() == 2:
                    c_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
                    c_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
                    c_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
                elif x.dim() == 3:  # Batch computation
                    c_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
                    c_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
                    c_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)
                else:
                    print("x.shape : ", x.shape)
                    raise ValueError("Incorrect number of dimensions")
                return c_xx - 2 * c_xy + c_yy
        else:
            def cost(x, y):
                if x.dim() == 2:
                    n = x.shape[0]
                    m = y.shape[0]
                    d = x.shape[1]

                    x = x.unsqueeze(1).expand(n, m, d)
                    y = y.unsqueeze(0).expand(n, m, d)
                    dist = torch.norm(x - y, p=1, dim=2)
                elif x.dim() == 3:
                    num_batch = x.shape[0]
                    n = x.shape[1]
                    m = y.shape[1]
                    d = x.shape[2]

                    x = x.unsqueeze(2).expand(num_batch, n, m, d)
                    y = y.unsqueeze(1).expand(num_batch, n, m, d)
                    dist = torch.norm(x - y, p=1, dim=3)
                else:
                    print("x.shape : ", x.shape)
                    raise ValueError("Incorrect number of dimensions")
                return dist
    return cost