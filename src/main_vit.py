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
from models.Vit import VisionTransformer
from transformers import ViTForImageClassification
from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_metric
from transformers import ViTFeatureExtractor


CFG = {
    'IMG_SIZE': 224,
    'MAX_EPOCH': 1000,
    'EARLY_STOP': 10,
    'LEARNING_RATE': 5e-4,
    'BATCH_SIZE': 16,
    'SEED': 41
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


seed_everything(CFG['SEED'])
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

#/////////////////////// SETTING DATA LABEL //////////////////////////////////////
df = pd.read_csv('/home/prml/chanyoung/fire detection/train_info.csv')

le = preprocessing.LabelEncoder()
df['class'] = le.fit_transform(df['class'].values)

id2label = {id: label for id, label in enumerate(df['class'].unique())}
label2id = {label: id for id, label in id2label.items()}

tv_df, test_df, _, _ = train_test_split(
    df, df['class'].values, test_size=0.1, random_state=CFG['SEED'])
train_df, val_df, _, _ = train_test_split(
    tv_df, tv_df['class'].values, test_size=0.25, random_state=CFG['SEED'])

print(
    f'Trains: [{len(train_df)}], Val: [{len(val_df)}], Test: [{len(test_df)}] ')

train_dataset = CNN_Dataset(
    img_paths=train_df['file_path'].values, labels=train_df['class'].values, img_size=CFG['IMG_SIZE'])
train_loader = DataLoader(
    train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CNN_Dataset(
    img_paths=val_df['file_path'].values, labels=val_df['class'].values, img_size=CFG['IMG_SIZE'])
val_loader = DataLoader(
    val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


#/////////////////////// SETTING MODEL //////////////////////////////////////
train_order = ['1']


training_args = TrainingArguments(
    output_dir="./vit-base-beans",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

feature_extractor = ViTFeatureExtractor.from_pretrained(
    'google/vit-base-patch16-224')

for net_name in train_order:
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                      num_labels=3,
                                                      id2label=id2label,
                                                      label2id=label2id)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss().to(device)

    model_save_path = f"/home/prml/chanyoung/fire detection/saved_model/efficient_{net_name}.pt"
    log_save_path = f"/home/prml/chanyoung/fire detection/logs/efficient_{net_name}/"

    #/////////////////////// SETTING SOLVER //////////////////////////////////////
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    print(train_results)
