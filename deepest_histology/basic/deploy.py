from typing import Optional
from pathlib import Path

import torch
import torchvision
import pandas as pd
from torch import nn
from sklearn import preprocessing
from tqdm import tqdm

from ..experiment import TilePredsDF
from ..utils import log_defaults
from .data import DatasetLoader
from .model import initialize_model


@log_defaults
def deploy(model: torch.nn.Module, target_label: str, test_df: pd.DataFrame, result_dir: Path, *,
           model_name: str = 'resnet',
           batch_size: int = 64,
           feature_extract: bool = False,
           **kwargs) -> TilePredsDF:

    num_classes = 2 #TODO does this matter here?
    _, input_size = initialize_model(model_name, num_classes, feature_extract=feature_extract,
                                     use_pretrained=True)
    le = preprocessing.LabelEncoder()
    labels_list = le.fit_transform(test_df[target_label])
    target_label_dict = dict(zip(le.classes_, range(len(le.classes_))))

    test_x = list(test_df.tile_path)
    test_y = le.fit_transform(test_df[target_label])

    args = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory' : False}

    test_set = DatasetLoader(
        test_x, test_y, transform=torchvision.transforms.ToTensor, target_patch_size=input_size)
    test_generator = torch.utils.data.DataLoader(test_set, **args)  # type: ignore

    criterion = nn.CrossEntropyLoss()

    epoch_loss, epoch_acc, pred_list = validate_model(model, test_generator, criterion)

    scores = {f'{target_label}_{key}': [item[index] for item in pred_list]
              for index, key in enumerate(list(target_label_dict.keys()))}

    scores = pd.DataFrame.from_dict(scores)

    df = pd.concat([test_df.reset_index(), scores], axis=1)

    return df


def validate_model(model, dataloaders, criterion):
    phase = 'test'

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    predList = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Iterate over data.
    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(phase == 'train'):
            #outputs = model(inputs)
            outputs = nn.Softmax(dim=1)(model(inputs))
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            predList = predList + outputs.tolist()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders.dataset)
    epoch_acc = running_corrects.double() / len(dataloaders.dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc, predList