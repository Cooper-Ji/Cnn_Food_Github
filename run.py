import numpy as np
from data import MyData
from Frame import CNN
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from visdom import Visdom
from pytorchtools import EarlyStopping

viz = Visdom()
train_root = 'food-11/training/'
validation_root = 'food-11/validation/'
model_path = './cnn.pth'
batchSize = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_set = MyData(train_root, 0)
train_Loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)

validation_set = MyData(validation_root, 1)
validation_Loader = DataLoader(validation_set, batch_size=batchSize, shuffle=True)

net = CNN()
net.to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
num_epoch = 60
train_step = 0
patience = 6

early_stopping = EarlyStopping(patience=patience, verbose=True)

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    train_step += 1
    val_acc = 0.0
    val_loss = 0.0

    net.train()
    for i, data in enumerate(train_Loader):
        optimizer.zero_grad()
        img, label = data       # 4 组
        img, label = img.to(device), label.to(device)
        label = label.squeeze()
        result = net(img)
        bitchLoss = loss(result, label)

        bitchLoss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(result.cpu().data.numpy(), axis=1) == label.cpu().numpy())
        train_loss += bitchLoss.item()

        # if i % 100 == 99:
        #    print('[%03d/%03d] Train Acc: %3.6f Loss: %3.6f' % (epoch + 1, num_epoch, train_acc / (i+1)*4,
        #                                                       train_loss / (i+1)*4))

    print('[%03d/%03d] Train Acc: %3.6f Loss: %3.6f' % (epoch+1, num_epoch, train_acc / train_set.__len__(),
                                                        train_loss / train_set.__len__()))

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_Loader):
            val_img, val_label = data
            val_img, val_label = val_img.to(device), val_label.to(device)
            val_label = val_label.squeeze()
            val_res = net(val_img)
            val_batch_loss = loss(val_res, val_label)

            val_acc += np.sum(np.argmax(val_res.cpu().data.numpy(), axis=1) == val_label.cpu().numpy())
            val_loss += val_batch_loss.item()

    val_loss_avg = val_loss / validation_set.__len__()
    early_stopping(val_loss_avg, net)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    viz.line([[train_loss / train_set.__len__()], [val_loss / validation_set.__len__()]], [train_step], win='Loss',
             update="append", opts=dict(title='train_loss&val_loss', legend=['train_loss', 'val_loss']))

    print('[%03d/%03d] Val Acc: %3.6f Val_Loss: %3.6f' % (epoch + 1, num_epoch, val_acc / validation_set.__len__(),
                                                            val_loss / validation_set.__len__()))








