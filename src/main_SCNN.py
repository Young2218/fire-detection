import albumentations as A
import pandas as pd
import numpy as np
import os
import random

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset.CNN_Dataset import CNN_Dataset

from models.DCNN.SCNN import SCNN

from solver import Solver

CFG = {
    'IMG_SIZE':128,
    'MAX_EPOCH':1000,
    'EARLY_STOP':10,
    'LEARNING_RATE':1e-5,
    'BATCH_SIZE':32,
    'MODEL_SAVE_PATH':"/home/bbb/ChanYoung/Fire_Detection/src/saved_model/SCNN.pth",
    'SEED':41
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#/////////////////////// SETTING DATA LABEL ////////////////////////////////////// 
df = pd.read_csv('/home/bbb/ChanYoung/Fire_Detection/src/data_info.csv')

le = preprocessing.LabelEncoder()
df['class'] = le.fit_transform(df['class'].values)

id2label = {id:label for id, label in enumerate(df['class'].unique())}
label2id = {label:id for id,label in id2label.items()}

tv_df, test_df, _, _ = train_test_split(df, df['class'].values, test_size=0.1, random_state=CFG['SEED'])
train_df, val_df, _, _ = train_test_split(tv_df, tv_df['class'].values, test_size=0.25, random_state=CFG['SEED'])

print(f'Trains: [{len(train_df)}], Val: [{len(val_df)}], Test: [{len(test_df)}] ')

train_dataset = CNN_Dataset(img_paths=train_df['file_path'].values, labels=train_df['class'].values, img_size=CFG['IMG_SIZE'])
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CNN_Dataset(img_paths=val_df['file_path'].values, labels=val_df['class'].values, img_size=CFG['IMG_SIZE'])
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

#/////////////////////// SETTING MODEL ////////////////////////////////////// 
model = SCNN(3)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'])
criterion = nn.CrossEntropyLoss().to(device)

#/////////////////////// SETTING SOLVER ////////////////////////////////////// 
solver = Solver(model=model,max_epoch=CFG['MAX_EPOCH'], early_stop=CFG['EARLY_STOP'],
                train_loader=train_loader, val_loader=val_loader, save_path=CFG['MODEL_SAVE_PATH'],
                optimizer=optimizer, criterion= criterion)
solver.train()