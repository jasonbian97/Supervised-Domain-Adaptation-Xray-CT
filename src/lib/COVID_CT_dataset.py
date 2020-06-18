
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir, self.classes[c], item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = image, int(self.img_list[idx][1])
        return sample

class MixedDataset(Dataset):
    def __init__(self, data_split, dataset_label, train = True,transform=None):
        if train:
            self.img_list = data_split[0]
            self.label_list = data_split[2]
            self.ds_label = dataset_label["train"]
        else:
            self.img_list = data_split[1]
            self.label_list = data_split[3]
            self.ds_label = dataset_label["val"]

        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = image, int(self.label_list[idx]), self.ds_label[idx]
        return sample


class CovidXray2clsDataset(Dataset):
    def __init__(self, data_split, train = True,transform=None):
        if train:
            self.img_list = data_split[0]
            self.label_list = data_split[2]
        else:
            self.img_list = data_split[1]
            self.label_list = data_split[3]

        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = image, int(self.label_list[idx])
        return sample
