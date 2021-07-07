import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, RobustScaler
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
    
def pre_processing(train_df):
        encode_map = {
    'Male': 1,'Female': 0,
    "Left":1,"Right":0,
    "G1":1,"G2":2,"G3":3,"G4":4,
    "0:LIVING":0,"1:DECEASED":1,
    "WHITE":0, "BLACK OR AFRICAN AMERICAN":1, 'ASIAN':2,
    'Elevated':0, 'Normal':1, 'Low':2,
    'WITH TUMOR':0, 'TUMOR FREE':1,
    'B0':0, 'B8':1,'BP':2,'CJ':3, 'CZ':4,'DV':5,'G6':6,
    'Yes':0, 'No':1,'YES':0,'NO':1,'Yes, History of Synchronous/Bilateral Malignancy':0,'Yes, History of Prior Malignancy':0,
    'T1a':0,'T1b':0,'T1c':0,'T1d':0,'T1':0,'T2a':1,'T2b':1,'T2c':1,'T2d':1,'T2':1,'T3a':2,'T3b':2,'T3c':2,'T3d':2,'T3':2,'T4a':3,'T4':3,'T4b':3,  
    }
        
        
 
        train_df =train_df.fillna(train_df['Hemoglobin level'].value_counts().index[0])
        train_df =train_df.fillna(train_df['Platelet count'].value_counts().index[0])
        train_df =train_df.fillna(train_df['Platelet count'].value_counts().index[0])
        train_df =train_df.fillna(train_df['WBC'].value_counts().index[0])
        train_df =train_df.fillna(train_df['Person Neoplasm Status'].value_counts().index[0])
        
        
        train_df['Sex'].replace(encode_map, inplace=True)
        train_df['Sex']=train_df['Sex'].astype(float)
        train_df["Race Category"].replace(encode_map, inplace=True)
        train_df["Race Category"]=train_df["Race Category"].astype(float)
        train_df["Primary Tumor Laterality"].replace(encode_map, inplace=True)
        train_df["Primary Tumor Laterality"]=train_df["Primary Tumor Laterality"].astype(float)
        train_df["Neoplasm Histologic Grade"].replace(encode_map, inplace=True)
        train_df["Neoplasm Histologic Grade"]=train_df["Neoplasm Histologic Grade"].astype(float)
        
        train_df["WBC"].replace(encode_map, inplace=True)
        train_df["WBC"]=train_df["WBC"].astype(float)
        train_df['Person Neoplasm Status'].replace(encode_map, inplace=True)
        train_df['Person Neoplasm Status']=train_df['Person Neoplasm Status'].astype(float)
        train_df["Tissue Source Site"].replace(encode_map, inplace=True)
        train_df["Tissue Source Site"]=train_df["Tissue Source Site"].astype(float)
        train_df['Platelet count'].replace(encode_map, inplace=True)
        train_df['Platelet count']=train_df['Platelet count'].astype(float)
        train_df['Primary Lymph Node Presentation Assessment Ind-3'].replace(encode_map, inplace=True)
        train_df['Primary Lymph Node Presentation Assessment Ind-3']=train_df['Primary Lymph Node Presentation Assessment Ind-3'].astype(float)
        train_df['Prior Cancer Diagnosis Occurence'].replace(encode_map, inplace=True)
        train_df['Prior Cancer Diagnosis Occurence']=train_df['Prior Cancer Diagnosis Occurence'].astype(float)
        train_df['Hemoglobin level'].replace(encode_map, inplace=True)
        train_df['Hemoglobin level']=train_df['Hemoglobin level'].astype(float)
        train_df['American Joint Committee on Cancer Tumor Stage Code'].replace(encode_map, inplace=True)
        train_df['American Joint Committee on Cancer Tumor Stage Code']=train_df['American Joint Committee on Cancer Tumor Stage Code'].astype(float)
        scaler = RobustScaler()
        train_df[["Diagnosis Age", "Sex","Neoplasm Histologic Grade",'Platelet count',"Primary Tumor Laterality",'American Joint Committee on Cancer Tumor Stage Code','Hemoglobin level','Prior Cancer Diagnosis Occurence',"Tissue Source Site","Race Category",'Primary Lymph Node Presentation Assessment Ind-3',"WBC",'Person Neoplasm Status']] = scaler.fit_transform(train_df[["Diagnosis Age", "Sex","Neoplasm Histologic Grade",'Platelet count',"Primary Tumor Laterality",'American Joint Committee on Cancer Tumor Stage Code','Hemoglobin level','Prior Cancer Diagnosis Occurence',"Tissue Source Site","Race Category",'Primary Lymph Node Presentation Assessment Ind-3',"WBC",'Person Neoplasm Status']])
        return train_df
    
def train(target_label: str, train_df: pd.DataFrame, result_dir, patience=2, **kwargs):
    train_df = train_df.sample(n=100)
    
    #preprocess stuff
    train_df = pre_processing(train_df)
    print(train_df)
    #extra_labels = ["Diagnosis Age", "Sex","Neoplasm Histologic Grade",'Platelet count',"Primary Tumor Laterality",'American Joint Committee on Cancer Tumor Stage Code','Hemoglobin level','Prior Cancer Diagnosis Occurence',"Tissue Source Site","Race Category",'Primary Lymph Node Presentation Assessment Ind-3',"WBC",'Person Neoplasm Status',]
    extra_labels = [("Diagnosis Age", RegressionBlock),("Neoplasm Histologic Grade",CategoryBlock),("Sex",CategoryBlock),("Platelet count",CategoryBlock),("Primary Tumor Laterality",CategoryBlock),("American Joint Committee on Cancer Tumor Stage Code",CategoryBlock),("Hemoglobin level",CategoryBlock),("Prior Cancer Diagnosis Occurence",CategoryBlock),("Tissue Source Site",CategoryBlock),("Race Category",CategoryBlock),("WBC",CategoryBlock),("Person Neoplasm Status",CategoryBlock),("Primary Lymph Node Presentation Assessment Ind-3",CategoryBlock)]#('Diagnosis Age', RegressionBlock),          "Neoplasm Histologic Grade","Platelet count","Primary Tumor Laterality",'American Joint Committee on Cancer Tumor Stage Code','Hemoglobin level','Prior Cancer Diagnosis Occurence',"Tissue Source Site","Race Category",'Primary Lymph Node Presentation Assessment Ind-3',"WBC",'Person Neoplasm Status' CategoryBlock)]#, 'Stromal Fraction']
    #extra_labels = ['year1']
    dblock = DataBlock(blocks=(ImageBlock, *(block for _, block in extra_labels), CategoryBlock),
                       getters=(ColReader('tile_path'),
                                *[ColReader(label) for label, _ in extra_labels],
                                ColReader(target_label)),
                       splitter=ColSplitter('is_valid'))
    
    #print(f'before: {len(train_df)}')
    #train_df = train_df.dropna(subset=extra_labels, how='any')
    #train_df = train_df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    #print(train_df)
    #train_df =train_df[extra_labels].fillna(df[extra_labels].value_counts().index[0],inplace=True)
    print(f'after: {len(train_df)}')
    
    dls = dblock.dataloaders(train_df, bs=4)
    
    learn = Learner(dls, LitClassifier(arch=resnet18, nf_additional=len(extra_labels), n_out=train_df[target_label].nunique()),
                    loss_func=CrossEntropyLossFlat())
    
    learn.fine_tune(2)
    learn.export(result_dir/'export.pkl')
    
    return learn

