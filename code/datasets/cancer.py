from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, data_dir, transform, data_type="train"):

        # Get Image File Names
        cdm_data = os.path.join(data_dir, data_type)  # directory of files

        file_names = os.listdir(cdm_data)  # get list of images in that directory
        self.full_filenames = [os.path.join(cdm_data, f) for f in file_names]  # get the full path to images

        # Get Labels
        labels_data = os.path.join(data_dir, "train_labels.csv")
        labels_df = pd.read_csv(labels_data)
        labels_df.set_index("id", inplace=True)  # set data frame index to id
        if data_type == "train":
            self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in file_names]  # obtained labels from df
        else:
            self.labels = None
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)  # size of dataset

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # Open Image with PIL
        image = self.transform(image)  # Apply Specific Transformation to Image
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image,


def cancer(params):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(20),
        transforms.Normalize((0.70244707, 0.54624322, 0.69645334), (0.23889325, 0.28209431, 0.21625058))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.70244707, 0.54624322, 0.69645334), (0.23889325, 0.28209431, 0.21625058))
    ])

    train_dataset = MyDataset(
        data_dir=params['data_dir'],
        transform=train_transform,
        data_type='train'
    )

    test_dataset = MyDataset(
        data_dir=params['data_dir'],
        transform=test_transform,
        data_type='test'
    )

    train_size = int(params['split']['train'] * len(train_dataset))
    valid_size = int(params['split']['valid'] * len(train_dataset))
    train_other_size = len(train_dataset) - train_size - valid_size

    train_datasets = random_split(
        train_dataset,
        [train_size, valid_size, train_other_size]
    )

    train_dataset, valid_dataset, _ = train_datasets

    return train_dataset, valid_dataset


