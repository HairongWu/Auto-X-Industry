# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from .autox_dataset import build

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

def build_dataset(datasetinfo):
    return build(datasetinfo)

