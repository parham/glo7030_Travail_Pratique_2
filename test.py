
import os

from torchviz import make_dot
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.models as models
import torchvision.datasets as ds
import torchvision.transforms as trans

# from pytorchvis.visualize_layers import VisualizeLayers

num_classes = 200

import question1 as q1

detect_device = lambda: torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_model():
    resnet18 = models.resnet18(pretrained=True)
    resnet18 = resnet18.cuda() if detect_device() else resnet18
    num_features = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_features, 200)
    resnet18.fc = resnet18.fc.cuda() if detect_device() else resnet18.fc

    return resnet18

trans_methods = trans.Compose([
    trans.Resize((224,224)),
    trans.ToTensor(),
    trans.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
])

ds_filename_valid = lambda filename : not os.path.basename(filename).startswith('.')

cub200 = ds.ImageFolder(root='./dataset/training', transform=trans_methods, is_valid_file=ds_filename_valid)

dl = DataLoader(cub200, batch_size=8, num_workers=2)

q1.separate_train_test('./dataset/training', './data_split/training', './data_split/testing')

resnet18 = models.resnet18(pretrained=True)

a = next(iter(dl))
out = resnet18(a[0])
vmodel = make_dot(out.mean(), params=dict(resnet18.named_parameters()))
vmodel.format = 'svg'
vmodel.render()
print(out)

# for d in dl:
#     out = resnet18(d[0])
#     vmodel = make_dot(out.mean(), params=dict(resnet18.named_parameters()))
#     vmodel.format = 'svg'
#     vmodel.render()
#     print(out)
#     break