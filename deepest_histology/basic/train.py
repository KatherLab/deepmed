import time
import logging
import os
from pathlib import Path

import torch
import torchvision
import pandas as pd
from torch import nn
from torch import optim
from sklearn import preprocessing
from tqdm import tqdm

from ..experiment import TrainDF
from ..utils import log_defaults
from .model import initialize_model
from .data import DatasetLoader


logger = logging.getLogger(__name__)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #TODO


@log_defaults
def train(target_label: str, train_df: TrainDF, result_dir: Path,
          model_name: str = 'resnet',
          batch_size: int = 64,
          freeze_ratio: float = .5,
          max_epochs: int = 10,
          opt: str = 'adam',
          lr: float = 1e-4,
          reg: float = 1e-5,
          **kwargs) -> torch.nn.Module:

    # preprocess data to fit old code
    num_classes = train_df[target_label].nunique()
    le = preprocessing.LabelEncoder()

    training_samples = train_df[~train_df.is_valid]
    train_x = list(training_samples.tile_path)
    train_y = torch.tensor(le.fit_transform(training_samples[target_label]), dtype=torch.long)

    valid_samples = train_df[train_df.is_valid]
    val_x = list(valid_samples.tile_path)
    val_y = torch.tensor(le.fit_transform(valid_samples[target_label]), dtype=torch.long)

    # get model / datasets
    model_ft, input_size = initialize_model(
        model_name, num_classes, feature_extract=False, use_pretrained=True)

    args = {'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory' : False}

    train_set = DatasetLoader(
        train_x, train_y, transform=torchvision.transforms.ToTensor, target_patch_size=input_size)
    traingenerator = torch.utils.data.DataLoader(train_set, **args) # type: ignore

    val_set = DatasetLoader(
        val_x, val_y, transform=torchvision.transforms.ToTensor, target_patch_size=input_size)
    valgenerator = torch.utils.data.DataLoader(val_set, **args) # type: ignore

    logger.debug(f'{device = }')
    model_ft = model_ft.to(device)

    no_of_layers = 0
    for name, child in model_ft.named_children():
        no_of_layers += 1

    cut = int(freeze_ratio * no_of_layers)

    ct = 0
    for name, child in model_ft.named_children():
        ct += 1
        if ct < cut:
            for name2, params in child.named_parameters():
                params.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optim(model_ft, opt=opt, lr=lr, reg=reg, params=False)
 
    model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = \
        train_model_classic(
            model=model_ft, trainLoaders=traingenerator, valLoaders=valgenerator,
            criterion=criterion, optimizer=optimizer, num_epochs=max_epochs,
            is_inception=(model_name == "inception"), results_dir=result_dir)

    df = pd.DataFrame(
        list(zip(train_loss_history, train_acc_history, val_loss_history, val_acc_history)),
        columns=['train_loss_history', 'train_acc_history', 'val_loss_history', 'val_acc_history'])
    df.to_csv(result_dir/'history.csv', index=False)

    return model


def train_model_classic(model, trainLoaders, valLoaders=[], criterion=None, optimizer=None,
                        num_epochs=25, is_inception=False, results_dir=''):

    since = time.time()

    train_acc_history = []
    train_loss_history = []

    val_acc_history = []
    val_loss_history = []

    early_stopping = EarlyStopping(patience=20, stop_epoch=20, verbose=True)

    for epoch in range(num_epochs):
        phase = 'train'
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(trainLoaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if is_inception and phase == 'train':
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(trainLoaders.dataset)
        epoch_acc = running_corrects.double() / len(trainLoaders.dataset)

        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)

        logger.info(f'{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}')

        if valLoaders:
            phase = 'val'

            model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            predList = []

            # Iterate over data.
            for inputs, labels in tqdm(valLoaders):
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

            val_loss = running_loss / len(valLoaders.dataset)
            val_acc = running_corrects.double() / len(valLoaders.dataset)

            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            logger.info(f'{phase} loss: {val_loss:.4f} acc: {val_acc:.4f}')

            early_stopping(epoch, val_loss, model,
                           ckpt_name=os.path.join(results_dir, "checkpoint.pt"))
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')

    return model, train_loss_history, train_acc_history, val_acc_history, val_loss_history


class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50, verbose=False):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def get_optim(model, opt: str, lr: float, reg: float, params=False):
    if params:
        temp = model
    else:
        temp = filter(lambda p: p.requires_grad, model.parameters())

    if opt == "adam":
        return optim.Adam(temp, lr=lr, weight_decay=reg)
    elif opt == 'sgd':
        return optim.SGD(temp, lr=lr, momentum=0.9, weight_decay=reg)
    else:
        raise NotImplementedError