import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import cv2

'''
1. torchvision's Exist Datasets
    - FashionMNIST
    - ...
'''
def offical_exist_data(data_name='FashionMNIST', batch_size=256):
    '''
    Load data in torchvision
    data type:
        [0,1] torch.float 
    '''
    if data_name == 'FashionMNIST':
        train = torchvision.datasets.FashionMNIST(root='../../Datasets/FashionMNIST',train=True,download=True,
                                              transform=transforms.ToTensor())
        test = torchvision.datasets.FashionMNIST(root='../../Datasets/FashionMNIST',train=False,download=True,
                                              transform=transforms.ToTensor())
    # elif ...
    # ...
    
    train_iter = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    return train_iter, test_iter

'''
2. Self Defined Datasets
    - Load data in memory
    - Load image data in device
    - ...
'''

class Datasets(torch.utils.data.Dataset):
    '''
    Decorate data to load in the DataLoader later
    input:
        features, labels: 
            type: numpy.ndarray or list or torch.Tensor
    '''
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, index):
        feature = torch.Tensor(self.features[index])
        label = torch.LongTensor(self.labels[index])
        return feature, label
    def __len__(self):
        return len(self.features)

# define label map
def label_map(id_labels=[], cls_labels=[],data_path='../Datasets/data1'):
    '''
    this function can map label's id to class name as well as class name to label's id 
    '''
    cls_list = os.listdir(data_path)
    id_name=list(range(len(cls_list)))
    cls_name=cls_list
    if cls_labels == []:
        return [cls_name[int(i)] for i in id_labels]
    elif id_labels == []:
        return list(map(cls_name.index,cls_labels))
# Load image data in device
def load_local_data(data_path = '../Datasets/data1', img_size=(28,28), img_type='L'):
    '''
    data folder tree:
        data1:
            class(0):
                image1
                image2
                ...
            class(1):
                image1
                image2
                ...
            ...
            class(n):
                ...
                
    img_size: HxW int
    img_type: 
        use to convert image by PIL.Image.convert() function
        options:
            'L': gray
            'RGB': 3-channel image
    '''
    # class name list
    cls_list = os.listdir(data_path)
    # all files' names
    all_cls_files = []
    for i in range(len(cls_list)):
        all_cls_files.append(os.listdir(os.path.join(data_path, cls_list[i])))
    # all files' paths
    all_feature_path = []
    for cls_i in range(len(all_cls_files)):
        for feature_i in range(len(all_cls_files[cls_i])):
            all_feature_path.append(os.path.join(data_path, cls_list[cls_i], all_cls_files[cls_i][feature_i]))
    #### handle features ####
    # use PIL load image data [0-255] RGB HxWxC
    features_PIL=[]
    for path in all_feature_path:
        img = Image.open(path)
        # convert to img_type
        img = img.convert(img_type)
        # resize image
        img = img.resize(img_size,Image.ANTIALIAS)#  Image.ANTIALIAS 最高质量
        features_PIL.append(img)
    # transform to Tensor [0,1]
    ToTensor = transforms.ToTensor()
    features = list()
    for feature in features_PIL:
        features.append(ToTensor(feature))
    features = torch.stack(features,0)
    #### handle labels ####
    # load labels
    cls_label = []
    for i in range(len(cls_list)):
        cls_name = []
        cls_name.append(cls_list[i])
        cls_label += (cls_name*len(all_cls_files[i]))
        
    # if not LongTensor will get some bugs...
    id_label = torch.LongTensor(label_map(cls_labels=cls_label))
    
    return features, id_label

# split data to train data and test data
def data_split_shuffle(feature,label,split_scale=0.7,shuffle=True):
    '''
    split data to train data and test data
    we can set split_scale=1,shuffle=True, then there will be shuffle only,
    but pay attention with return terms, because test_feature and test_label will be null array
    '''
    index = np.arange(features.shape[0])
    np.random.shuffle(index)
    train_index = index[:int(len(index)*split_scale)]
    test_index = index[int(len(index)*split_scale):]
    train_feature = feature[train_index]
    test_feature = feature[test_index]
    train_label = label[train_index]
    test_label = label[test_index]
    return train_feature,train_label,test_feature,test_label

# final function

def self_define_datasets_all_in_one(features=None, labels=None, # Load data in memory
                                    data_path = '../Datasets/data1', # Load image data in device
                                    img_size=(28,28), img_type='L',
                                    batch_size=256,split_scale=0.7,shuffle=True):
    '''
    combine all utils of self define datasets,which includes:
        load data from memory
        load image data from device folder
    '''
    if features is None:
        features, labels = load_local_data(data_path, img_size, img_type)
    train_feature,train_label,test_feature,test_label = data_split_shuffle(features,labels,split_scale,shuffle)
    train_data = Datasets(train_feature,train_label)
    test_data = Datasets(test_feature,test_label)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_iter, batch_size=batch_size, shuffle=True)
    return train_iter, test_iter


# transform image data that can be used directily by model
def trans_data(img, img_size=(28,28), img_type='L'):
    '''
    transform image data that can be used directily by model
    input:
        img: opencv or PIL image
        img_size: default (28,28)
        img_type: 'L': gray(defualt)
                  'RGB': 3-channels img
    '''
    # if img is opencv type or numpy type transform it to PIL
    if isinstance(img, np.ndarray):
        # cv2's iamge type is BGR but PIL's image type is RGB
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    # convert to img_type
    img = img.convert(img_type)
    # resize image
    img = img.resize(img_size,Image.ANTIALIAS)
    ToTensor = transforms.ToTensor()
    image = ToTensor(img)
    return image