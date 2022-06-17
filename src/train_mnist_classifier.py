# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import io

class Net(nn.Module):
    def __init__(self, img_dim=(1,28,28), num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(img_dim[0], 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class CNN(nn.Module):
    def __init__(self, img_dim=(1,28,28), num_classes=10):
        super(CNN, self).__init__()
        assert img_dim[1] in [28,32]
        self.fe = torch.nn.Sequential(
            nn.Conv2d(img_dim[0], 32,3,1),  #26, 28
            nn.MaxPool2d(2,2),  # 13, 14
            nn.ReLU(),
            nn.Dropout2d(0.5), #0.5
            nn.Conv2d(32,64,3,1),   # 11, 12
            nn.MaxPool2d(2, 2), # 5, 6
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, 3, 1),   #3, 4
            nn.ReLU(),
            nn.Flatten()
        )
        if img_dim[1] == 28:
            self.cla = torch.nn.Sequential(
                nn.Linear(1152, 128),
                nn.ReLU(),
                nn.Dropout2d(0.5),
                nn.Linear(128, num_classes),
            )
        elif img_dim[1] == 32:
            self.cla = torch.nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(),
                nn.Dropout2d(0.5),
                nn.Linear(128, num_classes),

            )

    def forward(self, x):
        fes = self.fe(x)
        pred = self.cla(fes)
        return F.log_softmax(pred, dim=1)

    def pred(self, x):
        return F.softmax(self.cla(self.fe(x)), dim=1)

class MLP(nn.Module):

    def __init__(self, img_dim=(1,28,28), num_classes=10):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(np.prod(img_dim), 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=1)

    def pred(self, x):
        return F.softmax(self.net(x), dim=1)


class LogReg(nn.Module):

    def __init__(self, img_dim=(1,28,28), num_classes=10):
        super(LogReg, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(np.prod(img_dim), num_classes),
        )

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=1)

    def pred(self, x):
        return F.softmax(self.net(x), dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        #for p in model.parameters():
        #    p.grad =None
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    return
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     if args.dry_run:
        #         break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    outputs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            #data, target = data.to(device), target.to(device)
            output = model.pred(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().cpu().item()
            outputs.append(output)
    preds = torch.cat(outputs, dim=0)
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    return acc, preds
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

def torch_evaluate(train_img, train_label, val_image, val_label, device, arch='MLP', batch_size=512, patience=10, optim_choice='Adam'):
    """ hidden_layer_sizes=(100,), activation="relu", *,
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000"""
    img_dim = val_image.shape[1:]
    num_sample = train_img.shape[0]
    if arch == 'MLP':

        net = MLP(img_dim, num_classes=val_label.max().item()+1).cuda()
        train_img = train_img.reshape([num_sample, -1])
        val_image = val_image.reshape([val_image.shape[0], -1])
    elif arch == 'LogReg':
        net = LogReg(img_dim, num_classes=val_label.max().item()+1).cuda()
        train_img = train_img.reshape([num_sample, -1])
        val_image = val_image.reshape([val_image.shape[0], -1])
    else:
        net = CNN(img_dim, num_classes=val_label.max().item()+1).cuda()

    n_train = int(num_sample*0.9)
    trainloader = DataLoader(TensorDataset(train_img[:n_train].cuda(), train_label[:n_train].cuda()), shuffle=True,
                             batch_size=batch_size, num_workers=0)
    valloader = DataLoader(TensorDataset(train_img[n_train:].cuda(), train_label[n_train:].cuda()), batch_size=batch_size, num_workers=0)
    testloader = DataLoader(TensorDataset(val_image, val_label), batch_size=batch_size, num_workers=0)

    if arch in ['MLP', 'LogReg'] or optim_choice == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6)
        max_epoch = 300
    else:
        optimizer = optim.SGD(net.parameters(), lr=1e-3)
        max_epoch = 500

    patience = patience
    best_acc = 0
    last_improved = 0

    for e in range(max_epoch):
        train(net, None, trainloader, optimizer, None)
        a, _ = test(net, None, valloader)
        if a > best_acc or e == 0:
            best_acc = a
            buffer = io.BytesIO()
            torch.save(net.state_dict(), buffer)
            last_improved = 0
        else:
            last_improved += 1
        if last_improved >= patience:
            print('patience reached in %d epochs' % e)
            break
    buffer.seek(0)
    net.load_state_dict(torch.load(buffer))
    a, preds = test(net, None, testloader)
    return a, preds

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    # dataset1 = datasets.MNIST('datasets/mnist', train=True, download=False,
    #                     transform=transform)
    # dataset2 = datasets.MNIST('datasets/mnist', train=False,
    #                    transform=transform)
    from .data.CelebA import MyCelebA
    #dataset1 = datasets.FashionMNIST('datasets/fashion', train=True, download=False,
    #                    transform=transform)
    #dataset2 = datasets.FashionMNIST('datasets/fashion', train=False,
    #                   transform=transform)

    dataset1 = MyCelebA('C:/Users/jean/celeb', split='train', download=False,
                                     transform=transform)
    dataset2 = MyCelebA('C:/Users/jean/celeb', split='train',
                                     transform=transform)
    net_type='cnn'
    if net_type == 'cnn':
        net = CNN((3,32,32)).to('cuda')
    else:
        net = MLP((3,32,32)).to('cuda')
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    patience = 10
    max_epoch = 200

    best_acc = 0
    last_improved = 0
    import time
    #
    # from data import dataset_to_tensor
    # train_img, train_label = dataset_to_tensor(dataset1)
    # val_img, val_label = dataset_to_tensor(dataset2)
    #
    # if net_type == 'cnn':
    #     train_loader = DataLoader(TensorDataset(train_img.cuda(), train_label.cuda()),
    #                               shuffle=True,
    #                               batch_size=512, num_workers=0)  # .reshape(train_img.shape[0], -1).cuda()
    #     test_loader = DataLoader(TensorDataset(val_img.cuda(), val_label.cuda()),
    #                              batch_size=512, num_workers=0)  # .reshape(val_img.shape[0], -1)
    # else:
    #     train_loader = DataLoader(TensorDataset(train_img.reshape(train_img.shape[0], -1).cuda(), train_label.cuda()), shuffle=True,
    #                                  batch_size=512, num_workers=0) #.reshape(train_img.shape[0], -1).cuda()
    #     test_loader = DataLoader(TensorDataset(val_img.reshape(val_img.shape[0], -1).cuda(), val_label.cuda()),
    #                                 batch_size=512, num_workers=0) #.reshape(val_img.shape[0], -1)

    #from data import dataset_to_tensor
    #train_img, train_label = dataset_to_tensor(dataset1)
    #val_img, val_label = dataset_to_tensor(dataset2)


    train_loader = DataLoader(dataset1,
                              shuffle=True,
                              batch_size=512, num_workers=8)  # .reshape(train_img.shape[0], -1).cuda()
    test_loader = DataLoader(dataset2, batch_size=512, num_workers=8)  # .reshape(val_img.shape[0], -1)

    for e in range(max_epoch):
        t0 = time.clock()
        train(net, None, train_loader, optimizer, None)
        print(time.clock() - t0)
        a, _ = test(net, None, test_loader)
        print(a)
        if a > best_acc:
            best_acc = a
            buffer = io.BytesIO()
            torch.save(net.state_dict(), buffer)
            last_improved = 0
        else:
            last_improved += 1
        if last_improved >= patience:
            print('patience reached in %d epochs' % e)
            break
    buffer.seek(0)
    net.load_state_dict(torch.load(buffer))
    a, preds = test(net, None, test_loader)
    print(a)    #91.5% CNN

if __name__ == '__main__':
    main()