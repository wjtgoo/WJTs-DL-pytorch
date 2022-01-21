import matplotlib.pyplot as plt
import torch
import numpy as np

device = torch.device('cpu')

# show imges and labels
def show_images_labels(images, labels):
    '''
    show images and labels in a line
    input: 
        images: torch.Tenosr [0,1] nxCxHxW
        labels: list or Tensor or ndarray
    '''
    # put data into cpu
    if isinstance(images, torch.Tensor):
        images.to(device)
    else:
        raise Exception('input images\' type not Tensor ')
    if isinstance(labels, torch.Tensor):
        labels.to(device)
        labels = labels.tolist()
    _, figs = plt.subplots(1, len(images), figsize=(12,12))
    for f, img, lbl in zip(figs, images, labels):
        img_c_h_w = img.numpy()
        img_c_h_w = np.squeeze(img_c_h_w)
        if len(img_c_h_w.shape)==3:
            img_h_w_c = np.transpose(img_c_h_w, (1,2,0))
        else:
            img_h_w_c = img_c_h_w
        f.imshow(img_h_w_c)
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# show images real labels and predicted labels
def show_images_real_pre_labels(images, real_labels, pre_labels):
    '''
    show images and labels in a line
    input: 
        images: torch.Tenosr [0,1] nxCxHxW
        labels: list or Tensor or ndarray
    '''
    # put data into cpu
    if isinstance(images, torch.Tensor):
        images.to(device)
    else:
        raise Exception('input images\' type not Tensor ')
    if isinstance(real_labels, torch.Tensor):
        real_labels.to(device)
        real_labels = real_labels.tolist()
    if isinstance(pre_labels, torch.Tensor):
        pre_labels.to(device)
        pre_labels = pre_labels.tolist()
        
    _, figs = plt.subplots(1, len(images), figsize=(12,12))
    for f, img, rlbl, plbl in zip(figs, images, real_labels, pre_labels):
        img_c_h_w = img.numpy()
        img_c_h_w = np.squeeze(img_c_h_w)
        if len(img_c_h_w.shape)==3:
            img_h_w_c = np.transpose(img_c_h_w, (1,2,0))
        else:
            img_h_w_c = img_c_h_w
        f.imshow(img_h_w_c)
        f.set_title('real label:{}\npredict label:{}'.format(rlbl,plbl))
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
    
# show train status
def training_status_per_epoch(train_loss, train_acc, test_acc):
    train_acc = train_acc.copy()
    test_acc = test_acc.copy()
    train_acc.insert(0,0)
    test_acc.insert(0,0)
    x = np.arange(len(train_loss)+1)
    xl = np.arange(len(train_loss))+1
    fig = plt.figure()
    ax_left = fig.add_subplot(111)
    lins1 = ax_left.plot(x, train_acc,label='train acc')
    lins2 = ax_left.plot(x, test_acc,label='test acc')
    ax_left.set_xlabel('epoch')
    ax_left.set_ylabel('Accuracy')
    ax_right = ax_left.twinx()
    lins3 = ax_right.plot(xl, train_loss,'r',label='train loss')
    ax_right.set_ylabel('Loss')
    ax_right.set_xlabel('epoch')
    lins = lins1+lins2+lins3
    labs = [l.get_label() for l in  lins]
    ax_left.legend(lins, labs, loc='center right')
    plt.show()