#!/usr/bin/env python3
"""Extracts features with a custom model."""
from deepmed.experiment_imports import *
from deepmed.get._extract_features import Extract
import torch
import torchvision


# The below code is taken from
# <https://github.com/ozanciga/self-supervised-histopathology>
def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weights could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model


model = torchvision.models.__dict__['resnet18'](pretrained=False)
state = torch.load('tenpercent_resnet18.ckpt', map_location='cuda:0')

state_dict = state['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace('model.', '').replace(
        'resnet.', '')] = state_dict.pop(key)

model = load_model_weights(model, state_dict)


# assuming `model` contains a model of choice, calling `Extract` with
# `arch=lambda pretrained: model` will extract features with that model.
def main():
    do_experiment(
        project_dir='.',
        get=Extract(
            feat_dir='features/ozanciga',
            tile_dir='tile/dir',
            arch=lambda pretrained: model,
        ),
        devices={'cuda:0': 4}
    )


if __name__ == '__main__':
    main()
