import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from keras import layers, regularizers, optimizers

from keras.utils.np_utils import to_categorical, normalize
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback, EarlyStopping, LambdaCallback
from keras.initializers import glorot_normal, RandomNormal, Zeros
from keras.models import Model, load_model
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import GaussianNoise

from skimage.transform import rotate
from skimage.util import random_noise


from google.colab import drive
from google.colab import files
from __future__ import print_function
import pickle as pkl
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cv2

"""## Data Preprocess

### import dataset
"""

label="/content/drive/My Drive/551A3/dataset/labels_l.pkl"
images="/content/drive/My Drive/551A3/dataset/images_l.pkl"
test="/content/drive/My Drive/551A3/dataset/images_test.pkl"
images_ul="/content/drive/My Drive/551A3/dataset/images_ul.pkl"

val=27000
batch=120


with open(images, 'rb') as f: 
  x_input = pickle.load(f) ## Image data for training (30,000 sample, each sample is a 56x56 image)

  x_train=x_input[:val]
  x_train = x_train.astype(np.float32)
  x_val=x_input[val:]
  x_val = x_val.astype(np.float32)


with open(label, 'rb') as f: ## Labels for training (30,000 rows, each row is a size 36 binary vector, which is the label to the corresponding image)
  y_input = pickle.load(f) 
  y_train = y_input[:val]
  y_train = y_train.astype(np.float32)
  y_val=y_input[val:]
  y_val = y_val.astype(np.float32)
  
 
with open(test, 'rb') as f: 
  x_test = pickle.load(f) ## Test images. The prediction corresponding to these images should be uploaded. (15,000 samples)
  x_test = x_test.astype(np.float32)
  x_test_loader=torch.utils.data.DataLoader(x_test, batch_size=batch, shuffle=False)

with open(images_ul, 'rb') as f: 
  images_ul = pickle.load(f) ##Additional images that can be used for training the classifier. Labels for these images are not provided. (30,000 samples, where each sample is a 56x56 image)
  images_ul = np.reshape(images_ul, (30000, 1, 56, 56))

"""### Data Augmentation"""

final_train_data = []
final_target_train = []
print(type(x_train))
for i in range(x_train.shape[0]):
    final_train_data.append(x_train[i])
    final_train_data.append(rotate(x_train[i], angle=45, mode = 'constant'))
    final_train_data.append(np.fliplr(x_train[i]).copy())
    final_train_data.append(np.flipud(x_train[i]).copy())
    final_train_data.append(random_noise(x_train[i],var=0.2**2).copy())
    for j in range(5):
        final_target_train.append(y_train[i])
print(len(final_train_data))

final_target_train = np.array(final_target_train).astype(np.float32)
final_train_data = np.array(final_train_data).astype(np.float32)
x_train_loader = torch.utils.data.DataLoader(final_train_data, batch_size=batch, shuffle=False)
y_train_loader = torch.utils.data.DataLoader(final_target_train, batch_size=batch, shuffle=False)

x_val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch, shuffle=False)
y_val_loader = torch.utils.data.DataLoader(y_val, batch_size=batch, shuffle=False)

x_test_loader=torch.utils.data.DataLoader(x_test, batch_size=batch, shuffle=False)

"""### Transform Data"""

from torch.utils.data import Dataset

class MNISTDataset(Dataset):
  def __init__(self, x, y, transform = transforms.Compose([
                                                           transforms.ToPILImage(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.5,), std=(0.5,))])
  ):
        self.transform=transform
        self.x = x[:,:,:,None]
        self.y = y

  def __len__(self):
        return len(self.x)

  def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #idx = idx.tolist()
        x = self.transform(self.x[idx])
        # if self.y is not None:
        #   return x, self.y[idx]
        # else:
        return x.reshape(56,56)

####
# transform data


x_train_transformed = MNISTDataset(x_train,y_train)
x_val_transformed = MNISTDataset(x_val,y_val)

x_train = x_train_transformed
x_val = x_val_transformed

"""### Load Data"""

# load data
x_train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch, shuffle=False)
x_val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch, shuffle=False)
y_train_loader = torch.utils.data.DataLoader(y_train, batch_size=batch, shuffle=False)
y_val_loader = torch.utils.data.DataLoader(y_val, batch_size=batch, shuffle=False)
images_ul_loader=torch.utils.data.DataLoader(images_ul, batch_size=batch, shuffle=False)

"""### Show Image"""

#view image
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg[0])
    plt.show()

# get some random training images
dataiter = iter(x_train_loader)
images = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

"""### Show Plot"""

def showPlot(epoch_list, val_accuracy_list, train_accuracy_list, model_name):

    plt.plot(epoch_list, val_accuracy_list, label='Validation Accuracy') 
    plt.plot(epoch_list, train_accuracy_list, label='Training Accuracy')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f'Accuracy plot for {model_name} in {len(epoch_list)} epochs')
    plt.show()

"""## Build Models

### CNN Base Model
"""

# Define a convolutional neural network ！！！

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #try conv3
        self.conv3 = nn.Conv2d(16,32,5)
        self.fc1 = nn.Linear(32 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 36)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 3 * 3)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

"""### VGG"""

import torch
import torch.nn as nn
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 36)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def VGG11():
    return VGG('VGG11')

def VGG13():
    return VGG('VGG13')

def VGG16():
    return VGG('VGG16')

def VGG19():
    return VGG('VGG19')

"""### ResNet 34"""

#18/34
class BasicBlock(nn.Module):
    expansion = 1 #每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):#downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)#BN处理
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x #捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

#50,101,152
class Bottleneck(nn.Module):
    expansion = 4#4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,#输出*4
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=36, include_top=True):#block残差结构 include_top为了之后搭建更加复杂的网络
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)自适应
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=36, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

"""### DenseNet 121"""

'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=36):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(1, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

# def DenseNet169():
#     return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

# def DenseNet201():
#     return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

# def DenseNet161():
#     return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

# def densenet_cifar():
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

# def test_densenet():
#     net = densenet_cifar()
#     x = torch.randn(1,1,45,45)
#     y = net(Variable(x))
#     print(y.size())

"""## Train Model"""

# Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, the following should print a CUDA device
print(device)

#!!! epoch/val/train accuracy will be refreshed if model is restarted
epoch_global=0
epoch_list=[]
val_accuracy_list=[]
train_accuracy_list=[]

#resnet34 = resnet34(36).to(device)
vgg16 = VGG16().to(device)
# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg13.parameters(), lr=0.0001)
#optimizer = optim.SGD(vgg13.parameters(), lr=0.0001, momentum=0.9)

# Train the network
def TrainModel(epochs, model):
  for epoch in range(epochs):  # loop over the dataset multiple times

      running_loss = 0.0
      x=0
      for i,(inputs, labels) in enumerate(zip(x_train_loader,y_train_loader)):

          # get the inputs; data is a list of [inputs, labels]

          inputs=inputs.unsqueeze(1)
          inputs=inputs.to(device)
          labels=labels.to(device)
        # labels=labels.unsqueeze(1)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = model(inputs)

        #  outputs=outputs.unsqueeze(1)
          loss = criterion(outputs, labels)

          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 100 == 99:    # print every 10 mini-batches
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
              running_loss = 0.0

  print('Finished Training')

def output_conversion(matrix):
  #print(type(matrix))
  converted_output = np.zeros(matrix.shape)
  
  index1 = np.argmax(matrix[0:10])
  index2 = np.argmax(matrix[10:36])
  converted_output[index1] = 1
  converted_output[index2+10] = 1
  return converted_output

def tostring(converted_output):

  lab=""
  for item in converted_output:
    lab+=str(int(item))
  return lab

"""## Calculate Accuracy

### Training Accuracy
"""

def get_training_accuracy(model):
  correct = 0
  total = 0
  true_list=[]
  pred_list=[]

  with torch.no_grad():
      count=0
      right=0
      for i,(images, labels) in enumerate(zip(x_train_loader,y_train_loader)):
          images=images.unsqueeze(1).to(device)

          outputs = model(images)
          '''
          for output, trulab in zip(outputs.numpy(),labels.numpy()):
            print(output)
            print(trulab)
            if count==0:
              break
            lab_pred=tostring(output_conversion(output))
            lab_true=tostring(truelab)
            pred_list.append(lab_pred)
            count+=1
            true_list.append(lab_true)
            
          '''
          for x in range(batch):
            output=outputs.cpu().numpy()[x]
            truelab=labels.numpy()[x]

            lab_pred=tostring(output_conversion(output))
            lab_true=tostring(truelab)
            pred_list.append(lab_pred)
            count+=1
            true_list.append(lab_true)
            if lab_pred==lab_true:

              right+=1
            #print("x={},right={}".format(x,right))
  
  print(f'{right}/{count}')
  return (right/count)

  #print(np.count_nonzero(np.array(true_list) == np.array(pred_list)))

"""### Validation Accuracy"""

def get_validation_accuracy(model):
  correct = 0
  total = 0
  true_list=[]
  pred_list=[]

  with torch.no_grad():
      count=0
      right=0
      for i,(images, labels) in enumerate(zip(x_val_loader,y_val_loader)):
          images=images.unsqueeze(1).to(device)

          outputs = model(images)
          '''
          for output, trulab in zip(outputs.numpy(),labels.numpy()):
            print(output)
            print(trulab)
            if count==0:
              break
            lab_pred=tostring(output_conversion(output))
            lab_true=tostring(truelab)
            pred_list.append(lab_pred)
            count+=1
            true_list.append(lab_true)
            
          '''
          for x in range(batch):
            output=outputs.cpu().numpy()[x]
            truelab=labels.numpy()[x]

            lab_pred=tostring(output_conversion(output))
            lab_true=tostring(truelab)
            pred_list.append(lab_pred)
            count+=1
            true_list.append(lab_true)
            if lab_pred==lab_true:

              right+=1
            #print("x={},right={}".format(x,right))
          
  print(f'{right}/{count}')
  return (right/count)

"""## Save Result"""

def runTestSet(model):
  test_list=[]
  for i,images in enumerate(x_test_loader):
          images=images.unsqueeze(1).to(device)

          outputs = model(images)

          for x in range(batch):
            output=outputs.cpu().detach().numpy()[x]

            lab_pred=tostring(output_conversion(output))

            test_list.append(lab_pred)
  return test_list

def writeOutput(model_name,nEpoch,accuracy,test_list):
  id_list = list(range(15000))


  import csv
  from itertools import zip_longest
  Id = id_list
  Category = test_list
  data = [Id, Category]
  export_data = zip_longest(*data, fillvalue = '')
  fName = f'/content/drive/My Drive/551A3/submissions/{model_name}/{accuracy}_{nEpoch}epoch.csv'

  with open(fName, 'w', encoding="ISO-8859-1", newline='') as file:
        write = csv.writer(file)
        write.writerow(("# Id", "Category"))
        write.writerows(export_data)
  file.close()
  print(f'Save to file {fName}')

"""## MAIN

"""

Total_epochs=50
#model=resnet34
model=vgg13
model_name='VGG13_rotate_transform'




for epoch in range(Total_epochs):
  epoch_global+=1


  print(f'--{epoch_global}--')
  TrainModel(1,model) #get accuracy for every epoch
  acc_train = get_training_accuracy(model)
  acc_val = get_validation_accuracy(model)


  epoch_list.append(int(epoch_global))
  train_accuracy_list.append(acc_train)
  val_accuracy_list.append(acc_val)

  
  # in very 5 epoch, show plot and write result
  if (epoch_global%5==0):
    showPlot(epoch_list, val_accuracy_list, train_accuracy_list,model_name)

    if(acc_val>=0.96):
      test_list = runTestSet(model)
      writeOutput(model_name, epoch_global,round(acc_val,2),test_list)

"""# ------------- 
# Backup Codes 

"""

#x_train = x_train.reshape(30000, 3136)
#x_test = x_test.reshape(15000, 3136)
x_train = x_train.reshape(x_train.shape[0], 56, 56, 1)
x_test = x_test.reshape(x_test.shape[0], 56, 56, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("Training matrix shape", x_train.shape)
print("Testing matrix shape", x_test.shape)

"""## build model"""

model = Sequential()
model.add(Dense(3136, input_shape=(3136,)))
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
                           
model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#model.add(Dense(36))
model.add(Dense(36, activation='sigmoid'))
#outputs = [Dense(36, activation='softmax', name=f"digit_{i}")(x) for i in range(2)]
#model.add(Activation('sigmoid')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probaility distribution, that is
                                 # that its values are all non-negative and sum to 1.
model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])



model.fit(x_train, y_train,
          batch_size=128, epochs=8, verbose=1,
          validation_data=(x_train, y_train))

def anothernet(input_dim=(56, 56, 1), out_dim=36):
    inputs = Input(shape=input_dim)

    x = BatchNormalization()(inputs)     
    x = GaussianNoise(0.1)(x)  
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)  
    x = Dropout(.2)(x)

    x = BatchNormalization()(x)  
    x = GaussianNoise(0.1)(x)  
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)      
    x = Dropout(.2)(x)

    x = BatchNormalization()(x)
    x = GaussianNoise(0.1)(x)  
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)    
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)  
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)   
    x = MaxPooling2D(pool_size=(2,2))(x) 
    x = Dropout(.25)(x) 

    x = BatchNormalization()(x)
    x = GaussianNoise(0.1)(x)  
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)  
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)    
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)  
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)  
    x = MaxPooling2D(pool_size=(2,2))(x)  
    x = Dropout(.25)(x) 
    x = GaussianNoise(0.05)(x)

    x = BatchNormalization()(x)
    x = Flatten()(x)   
    x = Dense(512,activation='relu')(x)
    x = Dropout(.5)(x)
    x = GaussianNoise(0.05)(x)
    outputs = [Dense(36, activation='sigmoid', name=f"digit_{i}")(x) for i in range(2)]
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model_test=anothernet()
model_test.fit(x_train, y_train,
          batch_size=512, epochs=1, verbose=1,
          validation_data=(x_train, y_train))

"""## performance evaluation"""

#helper function to convert prob to one hot encoding
def output_conversion(matrix):
  converted_output = np.zeros(matrix.shape)
  for i in range(matrix.shape[0]):
    max1 = np.max(matrix[i,0:9])
    index1 = matrix[i,0:9].tolist().index(max1)
    max2 = np.max(matrix[i,10:35])
    index2 = matrix[i,10:35].tolist().index(max2)
    converted_output[i,index1] = 1
    converted_output[i,index2+10] = 1
  return converted_output

preds = model.predict(x_train)
print(preds)
preds.shape

converted_predicted = output_conversion(preds)
converted_predicted_string=[]
y_train_string=[]
x=0
for item in converted_predicted:
    lab=""
    for label in item:
        lab+=str(int(label))

    converted_predicted_string.append(lab)

for it in y_train:
    lab=""
    for label in it:
        lab+=str(int(label))
    y_train_string.append(lab)
print(y_train_string[0])
print(converted_predicted_string[0])
acc_train = np.mean(converted_predicted_string == y_train_string)
print(acc_train)

"""## testset prediction"""

preds_test = model.predict(x_test)
converted_preds_test = output_conversion(preds_test)
print(converted_preds_test)
converted_preds_test.shape
print(converted_preds_test[0])

label_list=[]
id_list = list(range(15000))
x=0
for item in converted_preds_test:
    label=""
    for element in item:
        label+=str(int(element))
    label_list.append(label)
    if x == 0:
        print(label)
    x=x+1
print(id_list)

import csv
from itertools import zip_longest
Id = id_list
Category = label_list
data = [Id, Category]
export_data = zip_longest(*data, fillvalue = '')
with open('items.csv', 'w', encoding="ISO-8859-1", newline='') as file:
      write = csv.writer(file)
      write.writerow(("Id", "Category"))
      write.writerows(export_data)

file.close()

