'''犬猫データ分類'''
import os

import torch
from torch import nn
from torch.optim import Adam
from torch.utils import data

from torchvision import datasets, transforms

# from mymodel import Network


dirpath = os.path.dirname(__file__)

# 前処理
transform = transforms.Compose([transforms.ToTensor()])

# アノテーション
datapath = os.path.join(dirpath, 'dog_cat_data/train')
train_data = datasets.ImageFolder(datapath, transform)
valid_data = [train_data[x] for x in range(50)]
train_data = [train_data[x] for x in range(50, len(train_data))]


# model
# model = Network()
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=2)
print(model)

# loss, optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

batch_size = 10
num_epochs = 50

# DataLoaderを作成
train_dataloader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False)

dataloaders_dict = {
    'train': train_dataloader,
    'valid': valid_dataloader,
}

for epoch in range(num_epochs):
    print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
    print('-------------')

    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        epoch_loss = 0.0
        epoch_corrects = 0

        # DataLoaderからデータをバッチごとに取り出す
        for i, (inputs, labels) in enumerate(dataloaders_dict[phase]):
            print("\r" + "count:{}/{}".format(i+1, len(dataloaders_dict[phase])), end="")
            # optimizerの初期化
            optimizer.zero_grad()

            # trainの時は学習させる
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print(' {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
