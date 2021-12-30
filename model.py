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

dirpath = os.path.dirname(__file__)

# 前処理
transform = transforms.Compose([transforms.ToTensor()])

# アノテーション
datapath = os.path.join(dirpath, 'dog_cat_data/train')
train_data = datasets.ImageFolder(datapath, transform)

# 推論時の前処理
testpath = os.path.join(dirpath, 'dog_cat_data/test')
testpaths = natsorted(glob('{}/*.jpg'.format(testpath)))

# images = []
# for path in paths:
#     img = Image.open(path)
#     images.append(transform(img))
test_data = [transform(Image.open(path)) for path in testpaths]
test_data = torch.stack(test_data)


# model
# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=128 * 56 * 56, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = x.view(-1, 128 * 56 * 56)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x


# model = Network()
# Instantiate a neural network model
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=2)
print(model)

# loss, optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# バッチサイズの指定
batch_size = 10

# DataLoaderを作成
train_dataloader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

valid_dataloader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False)

# 辞書にまとめる
dataloaders_dict = {
    'train': train_dataloader,
    'valid': valid_dataloader,
}

# 動作確認
# イテレータに変換
batch_iterator = iter(dataloaders_dict['train'])

# 1番目の要素を取り出す
inputs, labels = next(batch_iterator)

# print(inputs.size())
# print(labels)


# エポック数
num_epochs = 50

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-------------')

    # for phase in ['train', 'val']:
    for phase in ['train']:

        if phase == 'train':
            # モデルを訓練モードに設定
            model.train()
        else:
            # モデルを推論モードに設定
            model.eval()

        # 損失和
        epoch_loss = 0.0
        # 正解数
        epoch_corrects = 0

        count = 0
        # DataLoaderからデータをバッチごとに取り出す
        for inputs, labels in dataloaders_dict[phase]:
            count += 1
            print("\r" + "count:{}/{}".format(count, len(dataloaders_dict[phase])), end="")
            # optimizerの初期化
            optimizer.zero_grad()

            # 学習時のみ勾配を計算させる設定にする
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                # 損失を計算
                loss = loss_fn(outputs, labels)
                # ラベルを予測
                _, preds = torch.max(outputs, 1)
                # 訓練時はバックプロパゲーション
                if phase == 'train':
                    # 逆伝搬の計算
                    loss.backward()
                    # パラメータの更新
                    optimizer.step()

                # イテレーション結果の計算
                # lossの合計を更新
                # PyTorchの仕様上各バッチ内での平均のlossが計算される。
                # データ数を掛けることで平均から合計に変換をしている。
                # 損失和は「全データの損失/データ数」で計算されるため、
                # 平均のままだと損失和を求めることができないため。
                epoch_loss += loss.item() * inputs.size(0)

                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
