# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from .autox_dataset import build


def build_dataset(datasetinfo):
    return build(datasetinfo)

