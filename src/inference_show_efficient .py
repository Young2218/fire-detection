import albumentations as A
import pandas as pd
import numpy as np
import os
import random
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset.CNN_Dataset import CNN_Dataset

from models.EfficientNet import EfficientNet
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, accuracy_score

CFG = {
    'IMG_SIZE':224,
    'MAX_EPOCH':1000,
    'EARLY_STOP':0,
    'LEARNING_RATE':5e-4,
    'BATCH_SIZE':1,
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
print(device)
#/////////////////////// SETTING DATA LABEL ////////////////////////////////////// 
test_df = pd.read_csv('/home/prml/chanyoung/fire-detection/custom_test_info.csv')

le = preprocessing.LabelEncoder()
test_df['class'] = le.fit_transform(test_df['class'].values)

id2label = {id:label for id, label in enumerate(test_df['class'].unique())}
label2id = {label:id for id,label in id2label.items()}


test_dataset = CNN_Dataset(img_paths=test_df['file_path'].values, labels=test_df['class'].values, img_size=CFG['IMG_SIZE'])
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


print(le.classes_)
print(le.transform(le.classes_))
#/////////////////////// SETTING MODEL //////////////////////////////////////
test_order = ['b0','b0','b1','b2','b3','b4','b5','b6','b7','v2_s','v2_m','v2_l']
# test_order = ['b0','b1','b2','b3','b4','b5']
summary = {}

for net_name in test_order:
    model = EfficientNet(num_class=3,model_name=net_name, pretrained=True)

    model_save_path = f"/home/prml/chanyoung/fire-detection/saved_model/small_training/efficient_{net_name}.pt"

    model.load_state_dict(torch.load(model_save_path))
    #/////////////////////// SETTING SOLVER ////////////////////////////////////// 


    model.eval()
    model.to(device)        
    gt_list = []
    pred_list = []
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            # get the inputs
            images = images.to(device)
            labels = labels.to(device)
            gt_list.append(labels.tolist()[0])
            
            output = model(images)
            output = torch.nn.functional.softmax(output[0], dim=0)
            confs, predicts = torch.topk(output, 1)
            predicts = predicts.to('cpu')
            pred_list.append(predicts.tolist()[0])
            
            
    # f1_score = sum(f1_list)/len(f1_score)
    print(net_name)
    #print(f'time: {time.time() - start_time}')
    print(f'accruacy: {accuracy_score(gt_list, pred_list)}')
    print(gt_list)
    print(pred_list)
    #print(confusion_matrix(gt_list, pred_list))
