import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
# import pytorch_lightning as pl
# from pytorch_lightning.logging import TensorBoardLogger
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from fastai.vision.all import *


class LitClassifier(nn.Module):
    def __init__(self, arch, nf_additional: int, n_out: int) -> None:
        super().__init__()        
        body = create_body(arch)
        self.cnn_feature_extractor = nn.Sequential(body, AdaptiveConcatPool2d(), nn.Flatten())
    
        nf_body = num_features_model(nn.Sequential(*body.children()))
        self.head = create_head(nf_body*2 + nf_additional, n_out, concat_pool=False)[3:]    # throw away pooling / flattenting layers
        
    def forward(self, img, *tab):
        img_feats = self.cnn_feature_extractor(img)
        #features = torch.cat((img_feats, tab), dim=1)
        stack_val = torch.stack((tab),axis=1)
        tensor_stack = cast(stack_val, Tensor)
        
        features = torch.cat([img_feats, tensor_stack], dim=1)
        return self.head(features)
    

def train(target_label: str, train_df: pd.DataFrame, result_dir, patience=2, **kwargs):
    train_df = train_df.sample(n=100)
    #extra_labels = ["Diagnosis Age", "Sex","Neoplasm Histologic Grade",'Platelet count',"Primary Tumor Laterality",'American Joint Committee on Cancer Tumor Stage Code','Hemoglobin level','Prior Cancer Diagnosis Occurence',"Tissue Source Site","Race Category",'Primary Lymph Node Presentation Assessment Ind-3',"WBC",'Person Neoplasm Status',]
    extra_labels = [('year1', RegressionBlock), ('gender', CategoryBlock)]#, 'Stromal Fraction']
    #extra_labels = ['year1']
    dblock = DataBlock(blocks=(ImageBlock, *(block for _, block in extra_labels), CategoryBlock),
                       getters=(ColReader('tile_path'),
                                *[ColReader(label) for label, _ in extra_labels],
                                ColReader(target_label)),
                       splitter=ColSplitter('is_valid'))
    
    print(f'before: {len(train_df)}')
    train_df = train_df.dropna(subset=extra_labels, how='any')
    #train_df = train_df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    #print(train_df)
    #train_df =train_df[extra_labels].fillna(df[extra_labels].value_counts().index[0],inplace=True)
    print(f'after: {len(train_df)}')
    
    dls = dblock.dataloaders(train_df, bs=24)
    
    learn = Learner(dls, LitClassifier(arch=resnet18, nf_additional=len(extra_labels), n_out=train_df[target_label].nunique()),
                    loss_func=CrossEntropyLossFlat())
    
    learn.fine_tune(1)
    learn.export(result_dir/'export.pkl')
    
    return learn

