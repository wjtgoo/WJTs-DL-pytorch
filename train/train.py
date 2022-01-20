import torch
from torch import nn
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_accuracy(data_iter, net, device=device):
    '''
    Evaluate the model accuracy in test datasets
    '''
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        net.eval() # 测试模式，此时会固定住dropout，BN的值
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
        net.train() # 改回训练模式
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, num_epochs, optimizer, loss=nn.CrossEntropyLoss(), device=device):
    '''
    Let's training
    return train_loss, train_acc, test_acc for every epoch
    '''
    net = net.to(device)
    print("training on ", device)
    train_loss_epochs, train_acc_epochs, test_acc_epochs = [], [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        train_loss_epochs.append(train_l_sum / batch_count)
        train_acc_epochs.append(train_acc_sum / n)
        test_acc_epochs.append(test_acc)
    return train_loss_epochs, train_acc_epochs, test_acc_epochs
