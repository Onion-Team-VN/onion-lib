import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import WineDataset
from .visualize import *
from .metrics import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    def __init__(self, n_feature, size_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, size_hidden)
        self.output = nn.Linear(size_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x


def train_2d(optimizer, steps=20):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        x1, x2, s1, s2 = optimizer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results


def train_engine_wine(optim_fn, hyperparams, batch_size=1, num_epoch=100, ylim=None):
    wine_dataset = WineDataset()
    train_loader = DataLoader(dataset=wine_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    n_feature = 13
    size_hidden = 200
    n_output = 3
    net = Net(n_feature=n_feature, size_hidden=size_hidden, n_output=n_output)

    optimizer = optim_fn(net.parameters(), **hyperparams)

    criterion = torch.nn.CrossEntropyLoss()

    ylim = [0.0, 0.5] if not ylim else ylim
    total_step = len(train_loader)
    animator = Animator(xlabel='epoch', ylabel='loss',
                        xlim=[0, num_epoch], ylim=ylim)
    loss_avg = AverageMeter()
    n = 0
    timer = Timer()
    for epoch in range(num_epoch):

        for i, (features, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            features = features.to(device)
            labels = labels.to(device)

            labels = labels.long()

            # Instructions: Forward pass and calculate loss
            outputs = net(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n += features.shape[0]
            loss_avg.update(loss.item())
            if n % 10 == 0:
                timer.stop()
                animator.add(n / batch_size / len(train_loader), loss_avg.avg)
                timer.start()
        if (epoch + 1) % (num_epoch / 20) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, {:.3f} sec/epoch'
                  .format(epoch + 1, num_epoch, i + 1, total_step, loss_avg.avg, timer.avg()))
    return timer.cumsum(), animator.Y[0]


def train_engine_lenet(net, train_set, num_epochs, lr, device):
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=8,
                                             shuffle=False)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc'])
    timer, num_batches = Timer(), len(train_iter)
    train_acc, train_loss = None, None
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(device), labels.to(device)
            predicted = net(features)
            loss = loss_func(predicted, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(loss * features.shape[0], accuracy(predicted, labels), features.shape[0])
            timer.stop()
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_loss, train_acc))
        print(f'epoch: {epoch}, loss {train_loss:.3f}, train acc {train_acc:.3f}')
        animator.add(epoch + 1, (None, None))

    return train_acc
