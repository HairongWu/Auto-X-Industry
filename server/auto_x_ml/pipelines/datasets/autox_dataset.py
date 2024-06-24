import torch
from skimage import io, transform
from torch.utils.data import Dataset

__all__ = ['build']

class AutoXDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, annotations, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.annotations[0][0]
        image = io.imread(img_name)
        anno = self.annotations[0][1]
        sample = {'image': image, 'anno': anno}

        if self.transform:
            sample = self.transform(sample)

        return sample

def build(datasetinfo):
    dataset = AutoXDataset(datasetinfo)
    return dataset

