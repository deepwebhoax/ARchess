import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import Dataset,DataLoader
import numpy as np 
import pandas as pd 
import shutil
import os
import zipfile
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import copy
import tqdm
from PIL import Image
from albumentations import pytorch as AT
import albumentations as A
import torchvision.datasets as dataset

import time

def test_transform():
    return A.Compose([
    A.Resize(96, 96),
     A.Normalize(),
    AT.ToTensor()
    ])
    
class TestDataset(Dataset):
    def __init__(self, image_list, transforms=None):
        super().__init__()
        self.image_list = image_list
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int):
        image = self.image_list[index]
        #image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image

class Detector():
    def __init__(self, args, device, model, testloader,classes):
        self.args = args
        self.model = model
        self.loader = testloader
        self.device = device
        self.classes=classes

    def predict(self):
        self.model.eval()
        desc=[]
        images = next(iter(self.loader))
        pred_mobilenet = np.array([])
        for batch_idx, (X_test, labels) in enumerate(self.loader):
            pred_mobilenet = np.append(pred_mobilenet, mobilenet.predict(X_test).tolist())
        print('Mobilenet prediction done!')
        return pred_mobilenet.reshape(8,8)

def createDataset(path,batch_size):
    test_dataset = TestDataset(path, test_transform())
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return test_loader


import torch
import torchvision
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import LRScheduler, Checkpoint 
from skorch.callbacks import Freezer, EarlyStopping

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

# for multiprocessing
import multiprocessing as mp
classes={0:'Empty',1:'whitePawn',2:'whiteBishop',3:'whiteKnight',4:'whiteRook',5:'whiteQueen',6:'whiteKing',7:'blackPawn',8:'blackBishop',9:'blackKnight',10:'blackRook',11:'blackQueen',12:'blackKing'}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args_dict = {'weights':'chess_weights/model_1.pt'}

class MobileNet(nn.Module):
    def init(self, output_features, num_units=512, drop=0.5,
                 num_units1=256, drop1=0.2):
        super().init()
        model =  torchvision.models.mobilenet_v2(pretrained=False)
        n_inputs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
                                nn.Dropout(p=drop1), 
                                nn.Linear(n_inputs, output_features))
        self.model = model
        
    def forward(self, x):
        return self.model(x)


lr_scheduler_mobilenet = LRScheduler(policy='StepLR',
                                  step_size=8,gamma=0.2)
# callback for saving the best on validation accuracy model
checkpoint_mobilenet = Checkpoint(f_params='/content/drive/MyDrive/chess_weights/best_model_mobilenet.pkl',
                            monitor='valid_acc_best')
# callback for freezing all layer of the model except the last layer
#freezer_vgg = Freezer(lambda x: not x.startswith('model.classifier'))
# callback for early stopping
early_stopping_mobilenet = EarlyStopping(patience=10)
mobilenet = NeuralNetClassifier(
    # pretrained ResNet50 + custom classifier 
    module=MobileNet,          
    # fine tuning model's inner parameters
    module__output_features=13,
    module__num_units=512,
    module__drop=0.5,
    module__num_units1=512,
    module__drop1=0.5,
    # criterion
    criterion=nn.CrossEntropyLoss,
    # batch_size = 128
    batch_size=20,
    # number of epochs to train
    max_epochs=100,
    # optimizer Adam used
    optimizer=torch.optim.Adam,
    optimizer__lr = 0.0025,
    optimizer__weight_decay=1e-6,
    # shuffle dataset while loading
    iterator_train__shuffle=True,
    # load in parallel
    iterator_train__num_workers=4,
    # stratified kfold split of loaded dataset
    train_split=CVSplit(cv=5, stratified=True, random_state=42),
    # callbacks declared earlier
    callbacks=[lr_scheduler_mobilenet, checkpoint_mobilenet, 
                early_stopping_mobilenet],
    # use GPU or CPU
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

mobilenet.initialize()

mobilenet.load_params(f_params='chess_weights/best_model_mobilenet.pkl')
# mobilenet.initialize()
# mobilenet.load_params(f_params='chess_weights/best_model_mobilenet.pkl')


# model = torchvision.models.mobilenet_v3_large(pretrained=True,progress=True)

def predict(tilePicturesArray):
    test_loader = createDataset(tilePicturesArray, 70)
    detector = Detector(args_dict,device,mobilenet,test_loader,classes)
    desc = detector.predict()
    return desc

if __name__=='__main__':
    from chessboard_detection import loadImage, inference
    from crop import cropChessboard
    from ndar_to_board import ndarr_to_board, RealPieces, minimax, moveToIndex
    path = 'input/31.jpg'
    imGrey = loadImage(path)  # loading for inference function
    im = Image.open(path)     # loading beauty
    
    tiles = inference(imGrey) # getting tiles coordinates
    tilePicturesArray = cropChessboard(np.array(im), tiles)
    print(predict(tilePicturesArray))