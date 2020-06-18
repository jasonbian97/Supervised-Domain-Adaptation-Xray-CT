
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
        self.txt_path = [txt_NonCOVID, txt_COVID ]
        self.classes = ['CT_NonCOVID', 'CT_COVID']
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


class CovidXray2clsDataset(Dataset):
    def __init__(self, dataset_info, train = True,transform=None):
        self.dataset_info = dataset_info
        data_split = self.split_train_val()
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

    def split_train_val(self):
        import sklearn.model_selection
        noncovid_img_list = [os.path.join(self.dataset_info["noncovid"], p) for p in
                             os.listdir(self.dataset_info["noncovid"])]
        covid_img_list = [os.path.join(self.dataset_info["covid"], p) for p in
                          os.listdir(self.dataset_info["covid"])]
        img_list = noncovid_img_list + covid_img_list
        label = [0] * len(noncovid_img_list) + [1] * len(covid_img_list)
        data_split = sklearn.model_selection.train_test_split(img_list, label, test_size=0.3)
        return data_split

class MixedDataset(Dataset):
    def __init__(self, dataset_info, train = True,transform=None):
        ct_ds_info = dataset_info["COVID-CT"]
        xray_ds_info = dataset_info["COVID-Xray2cls"]
        if train:
            self.ct_ds = CovidCTDataset(root_dir=ct_ds_info["image_folder"],
                                  txt_COVID=ct_ds_info["data_split"]+'/COVID/trainCT_COVID.txt',
                                  txt_NonCOVID=ct_ds_info["data_split"]+'/NonCOVID/trainCT_NonCOVID.txt',
                                  transform=transform)
            self.xray_ds = CovidXray2clsDataset(xray_ds_info, train = train, transform=transform)
            print("# of CT training: ",len(self.ct_ds))
            print("# of xray training: ", len(self.xray_ds))
        else:
            self.ct_ds = CovidCTDataset(root_dir=ct_ds_info["image_folder"],
                                txt_COVID=ct_ds_info["data_split"]+'/COVID/valCT_COVID.txt',
                                txt_NonCOVID=ct_ds_info["data_split"]+'/NonCOVID/valCT_NonCOVID.txt',
                                transform=transform)
            self.xray_ds = CovidXray2clsDataset(xray_ds_info, train=train, transform=transform)

    def __len__(self):
        return len(self.ct_ds)+len(self.xray_ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx<len(self.ct_ds):
            sample = self.ct_ds.__getitem__(idx) + (0,) # add ds_label indicating where this sample come from
        else:
            sample = self.xray_ds.__getitem__(idx-len(self.ct_ds)) + (1,)

        return sample