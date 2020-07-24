# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/9/5 16:16
@Describe：

"""
import platform
import os
import warnings
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from PIL.Image import Image
import torchvision
from torchvision import transforms, datasets

if platform.system() == 'Window':
    from mnist import mnist_nn
else:
    import mnist_nn

warnings.filterwarnings('ignore')


def plot_img(image):
    # print(image)
    image = image.numpy()[0]
    # print(image)
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    plt.imshow(image, cmap='gray')
    plt.show()


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    else:
        model.eval()
        volatile = True

    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            # print('cuda 可用')
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)

        # print(data.detach().numpy())

        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        running_loss += F.nll_loss(output, target, size_average=False).item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)
    # print(accuracy)
    print(f'{phase} loss is {loss} and {phase} accuracy '
          f'is {running_correct}/{len(data_loader.dataset)}  {accuracy}')
    return loss, accuracy


if __name__ == '__main__':

    data_folder = ''
    if platform.system() == 'Windows':
        data_folder = 'D:/home/developer/TrainData/mnist/data'
    else:
        data_folder = '/home/yanjilong/DataSets/mnist/data'

    # 获取数据
    transformation = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_folder, train=True, transform=transformation, download=True)
    test_dataset = datasets.MNIST(data_folder, train=False, transform=transformation, download=True)

    # 按批次加载
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)  #
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # print('*********' * 8)
    # for imgs, labs in train_loader:
    #     print(imgs.size())  # torch.Size([32, 1, 28, 28])

    sample_data = next(iter(train_loader))
    plot_img(sample_data[0][1])

    # 创建网络模型
    model = mnist_nn.Mnist_Net()
    if torch.cuda.is_available():
        model.cuda()
    # 优化函数
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    data, target = next(iter(train_loader))
    if torch.cuda.is_available():
        output = model(Variable(data.cuda()))
    else:
        output = model(Variable(data))

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []

    for epoch in range(1, 20):
        epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_loader, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='training loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r', label='validation loss')
    plt.legend()
    plt.savefig('plt_img/train.png')

    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='train accuracy')
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, 'r', label='val accuracy')
    plt.legend()
    plt.savefig('plt_img/test.png')

''''''
