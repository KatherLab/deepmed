import logging
import math
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import RobustScaler
# import pytorch_lightning as pl
# from pytorch_lightning.logging import TensorBoardLogger
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from fastai.vision.all import *
from deepest_histology.basic.train import fit_from_checkpoint


class LitClassifier(nn.Module):
    def __init__(self, arch, nf_additional: int, n_out: int) -> None:
        super().__init__()
        body = create_body(arch)
        self.cnn_feature_extractor = nn.Sequential(body, AdaptiveConcatPool2d(), nn.Flatten())

        nf_body = num_features_model(nn.Sequential(*body.children()))
        self.head = create_head(nf_body*2 + nf_additional, n_out, concat_pool=False)[3:]    # throw away pooling / flattenting layers

    def forward(self, img, *tab):
        img_feats = self.cnn_feature_extractor(img)

        if tab:
            stack_val = torch.stack((tab),axis=1)
            tensor_stack = cast(stack_val, Tensor)

            features = torch.cat([img_feats, tensor_stack], dim=1)
        else:
            features = img_feats
        return self.head(features)

def lit_splitter(model):
    return [params(model.cnn_feature_extractor), params(model.head)]

@dataclass
class Normalize:
    mean: float
    std: float

    def __call__(self, x):
        x = float(x)
        return (x - self.mean)/self.std if not math.isnan(x) else 0


def train(
        target_label: str, train_df: pd.DataFrame, result_dir, patience=2, batch_size=96, lr=2e-3,
        ng=None, num_workers: int = 0, max_epochs: int = 4, monitor: str = 'valid_loss',
          tfms=aug_transforms(
              flip_vert=True, max_rotate=360, max_zoom=1, max_warp=0, size=224),
        **kwargs):

    cont_names = [
        #'Diagnosis Age']
    
       'original_shape_Elongation', 'original_firstorder_90Percentile',
        'original_firstorder_Entropy', 'original_firstorder_Kurtosis',
        'original_firstorder_Maximum', 'original_glcm_Autocorrelation', 'original_glcm_Imc2',
        'original_glcm_Idmn', 'original_glrlm_LongRunHighGrayLevelEmphasis',
        'original_glszm_LargeAreaEmphasis', 'original_glszm_SizeZoneNonUniformityNormalized',
        'wavelet-LLH_firstorder_Entropy', 'wavelet-LLH_glrlm_GrayLevelNonUniformityNormalized',
        'wavelet-LHL_glszm_LargeAreaEmphasis',
        'wavelet-LHL_gldm_SmallDependenceLowGrayLevelEmphasis', 'wavelet-LHH_firstorder_Entropy',
        'wavelet-LHH_glcm_Autocorrelation', 'wavelet-LHH_glrlm_GrayLevelNonUniformityNormalized',
        'wavelet-LHH_glrlm_HighGrayLevelRunEmphasis',
        'wavelet-LHH_glszm_GrayLevelNonUniformityNormalized',
        'wavelet-LHH_glszm_HighGrayLevelZoneEmphasis',
        'wavelet-LHH_glszm_SizeZoneNonUniformityNormalized', 'wavelet-LHH_glszm_ZoneEntropy',
        'wavelet-LHH_gldm_DependenceEntropy', 'wavelet-HLL_glcm_Idmn',
        'wavelet-HLL_glcm_InverseVariance', 'wavelet-HLL_glszm_SizeZoneNonUniformityNormalized',
        'wavelet-HLL_gldm_DependenceNonUniformityNormalized', 'wavelet-HLH_firstorder_Entropy',
        'wavelet-HLH_glcm_DifferenceEntropy', 'wavelet-HLH_glrlm_GrayLevelNonUniformityNormalized',
        'wavelet-HLH_glrlm_HighGrayLevelRunEmphasis',
        'wavelet-HLH_glszm_GrayLevelNonUniformityNormalized',
        'wavelet-HLH_glszm_HighGrayLevelZoneEmphasis',
        'wavelet-HLH_glszm_SizeZoneNonUniformityNormalized',
        'wavelet-HLH_glszm_SmallAreaLowGrayLevelEmphasis', 'wavelet-HLH_glszm_ZoneEntropy',
        'wavelet-HLH_gldm_DependenceEntropy', 'wavelet-HHL_glszm_SizeZoneNonUniformityNormalized',
        'wavelet-HHL_gldm_SmallDependenceLowGrayLevelEmphasis', 'wavelet-HHH_firstorder_Entropy',
        'wavelet-HHH_glcm_Autocorrelation', 'wavelet-HHH_glrlm_GrayLevelNonUniformityNormalized',
        'wavelet-HHH_glrlm_HighGrayLevelRunEmphasis',
        'wavelet-HHH_glrlm_RunLengthNonUniformityNormalized',
        'wavelet-HHH_glszm_GrayLevelNonUniformityNormalized',
        'wavelet-HHH_glszm_HighGrayLevelZoneEmphasis',
        'wavelet-HHH_glszm_SizeZoneNonUniformityNormalized', 'wavelet-HHH_glszm_SmallAreaEmphasis',
        'wavelet-HHH_glszm_ZoneEntropy', 'wavelet-HHH_gldm_DependenceEntropy']
    
    for col in cont_names:
        train_df[col] = train_df[col].astype(float)

    conts = [(label, TransformBlock(type_tfms=[Normalize(mean=mean, std=std), RegressionSetup()]))
             for label in cont_names
             for mean, std in [(train_df[label].mean(), train_df[label].std())]]
    '''
    cats = [
        ('American Joint Committee on Cancer Metastasis Stage Code', CategoryBlock(add_na=True)),
        ('American Joint Committee on Cancer Tumor Stage Code', CategoryBlock(add_na=True)),
        ('Hemoglobin level', CategoryBlock(add_na=True)),
        ('Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code', CategoryBlock(add_na=True)),
        ('Neoplasm Disease Stage American Joint Committee on Cancer Code', CategoryBlock(add_na=True)),
        ('Neoplasm Histologic Grade', CategoryBlock(add_na=True)),
        ('Person Neoplasm Status', CategoryBlock(add_na=True)),
        ('Platelet count', CategoryBlock(add_na=True)),
        ('Primary Lymph Node Presentation Assessment Ind-3', CategoryBlock(add_na=True)),
        ('Primary Tumor Laterality', CategoryBlock(add_na=True)),
        ('Prior Cancer Diagnosis Occurence', CategoryBlock(add_na=True)),
        ('Race Category', CategoryBlock(add_na=True)),
        ('Sex', CategoryBlock(add_na=True)),
        ('Tissue Source Site', CategoryBlock(add_na=True)),
        ('WBC', CategoryBlock(add_na=True, vocab=['Low', 'Normal', 'Elevated']))]
    '''
    dblock = DataBlock(blocks=(ImageBlock,
                               #*(block for _, block in cats),
                               *(block for _, block in conts),
                               CategoryBlock),
                       getters=(ColReader('tile_path'),
                                #*(ColReader(name) for name, _ in cats),
                                *(ColReader(name) for name, _ in conts),
                                ColReader(target_label)),
                       splitter=ColSplitter('is_valid'),
                       batch_tfms=tfms)

    dls = dblock.dataloaders(train_df, bs=batch_size, num_workers=num_workers)

    learn = Learner(
        dls, LitClassifier(arch=resnet18, nf_additional=len(conts)+len(cats),
                           n_out=train_df[target_label].nunique()),
        loss_func=CrossEntropyLossFlat(), metrics=[BalancedAccuracy(), RocAucBinary()],
        path=result_dir, splitter=lit_splitter)

    cbs = [
        SaveModelCallback(monitor=monitor, fname=f'best_{monitor}', reset_on_fit=False),
        SaveModelCallback(every_epoch=True, with_opt=True, reset_on_fit=False),
        EarlyStoppingCallback(
            monitor=monitor, min_delta=0.001, patience=patience, reset_on_fit=False),
        CSVLogger(append=True)]

    if (result_dir/'models'/f'best_{monitor}.pth').exists():
        fit_from_checkpoint(
            learn=learn, result_dir=result_dir, lr=lr/2, max_epochs=max_epochs, cbs=cbs,
            monitor=monitor, logger=logging)
    else:
        learn.fine_tune(epochs=max_epochs, base_lr=lr, cbs=cbs)

    learn.export('export.pkl')

    return learn
