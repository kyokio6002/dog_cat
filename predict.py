'''犬猫データ分類'''
from PIL import Image
import os
from torchvision import transforms, datasets
from glob import glob
from natsort import natsorted
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim import Adam
from torch.autograd import Variable


model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=2)
model.load_state_dict(torch.load('model.pth'))
print(model)

dirpath = os.path.dirname(__file__)
# 前処理
transform = transforms.Compose([transforms.ToTensor()])

# アノテーション
datapath = os.path.join(dirpath, 'dog_cat_data/train')
train_data = datasets.ImageFolder(datapath, transform)

# 推論時の前処理
testpath = os.path.join(dirpath, 'dog_cat_data/test')
testpaths = natsorted(glob('{}/*.jpg'.format(testpath)))
test_data = [transform(Image.open(path)) for path in testpaths]
test_data = torch.stack(test_data)

results = model(test_data)
_, pred = torch.max(results, 1)
dict = {0: '猫', 1: '犬'}
print(pred)
print(pred.shape)

for i, value in enumerate(pred):
    print("{}:{}".format(testpaths[i].split('/')[-1], dict[value.item()]))
