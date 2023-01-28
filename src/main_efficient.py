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

from models.EfficientNet import EfficientNet

from solver import Solver

CFG = {
    'IMG_SIZE':224,
    'MAX_EPOCH':1000,
    'EARLY_STOP':10,
    'LEARNING_RATE':5e-4,
    'BATCH_SIZE':64,
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
df = pd.read_csv('/home/prml/chanyoung/fire-detection/aihub_train_info.csv')

le = preprocessing.LabelEncoder()
df['class'] = le.fit_transform(df['class'].values)

id2label = {id:label for id, label in enumerate(df['class'].unique())}
label2id = {label:id for id,label in id2label.items()}

train_df, val_df, _, _ = train_test_split(df, df['class'].values, test_size=0.25, random_state=CFG['SEED'])

print(f'Trains: [{len(train_df)}], Val: [{len(val_df)}]')

train_dataset = CNN_Dataset(img_paths=train_df['file_path'].values, labels=train_df['class'].values, img_size=CFG['IMG_SIZE'])
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CNN_Dataset(img_paths=val_df['file_path'].values, labels=val_df['class'].values, img_size=CFG['IMG_SIZE'])
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


#/////////////////////// SETTING MODEL //////////////////////////////////////
# train_order = ['b0','b1','b2','b3','b4','b5','b6','b7','v2_s','v2_m','v2_l']
train_order = ['b0','b1','b2','b3','b4','b5','v2_s']
summary = {}

for net_name in train_order:
    model = EfficientNet(num_class=3,model_name=net_name, pretrained=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=3, gamma=0.66)

    model_save_path = f"/home/prml/chanyoung/fire-detection/saved_model/aihub_training/efficient_{net_name}.pt"
    log_save_path = f"/home/prml/chanyoung/fire-detection/logs/aihub_training/efficient_{net_name}/"
    
    #/////////////////////// SETTING SOLVER ////////////////////////////////////// 
    solver = Solver(model=model,max_epoch=CFG['MAX_EPOCH'], early_stop=CFG['EARLY_STOP'],
                    train_loader=train_loader, val_loader=val_loader, 
                    save_path=model_save_path, log_save_path=log_save_path,
                    optimizer=optimizer, criterion= criterion, scheduler=scheduler)
    summary[net_name] = solver.train()

print(summary)