
import pytorch_lightning as pl
from torch.optim import Adam
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
import torchvision.models as models
import json
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger
from sklearn.metrics import confusion_matrix,f1_score
import os
os.chdir(os.path.dirname(__file__)) # set current .py file as working directory
import sys
sys.path.insert(0,"../..")
# customized packages
from src.lib.dataset import *
from src.lib.helper_func import *

class Mixed_COVID_CT_Xray_Sys(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # do this to save all arguments in any logger (tensorboard)
        self.hparams = hparams

        with open(hparams.dataset_info) as fp:
            self.dataset_info = json.load(fp)
            self.dataset_info = self.dataset_info[hparams.dataset_name]

        # split train and val
        # self.data_split,self.dataset_label = self.split_train_val()

        self.model = models.densenet169(pretrained=True)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, hparams.num_class)
        self.init_weights(self.model.classifier)
        # self.model.load_state_dict(torch.load(hparams.pretrained_path))

    # def split_train_val(self):
    #     import sklearn.model_selection
    #
    #
    #     CTnoncovid_img_list = [os.path.join(self.dataset_info["CTnoncovid"], p) for p in
    #                          os.listdir(self.dataset_info["CTnoncovid"])]
    #     CTcovid_img_list = [os.path.join(self.dataset_info["CTcovid"], p) for p in
    #                         os.listdir(self.dataset_info["CTcovid"])]
    #     CTlabel = [0] * len(CTnoncovid_img_list) + [1] * len(CTcovid_img_list)
    #     CTsplit =  sklearn.model_selection.train_test_split(CTnoncovid_img_list + CTcovid_img_list , CTlabel,
    #                                                         test_size=0.3, random_state=66)
    #
    #     Xraynoncovid_img_list = [os.path.join(self.dataset_info["Xraynoncovid"], p) for p in
    #                          os.listdir(self.dataset_info["Xraynoncovid"])]
    #     Xraycovid_img_list = [os.path.join(self.dataset_info["Xraycovid"], p) for p in
    #                         os.listdir(self.dataset_info["Xraycovid"])]
    #     Xraylabel = [0] * len(Xraynoncovid_img_list) + [1] * len(Xraycovid_img_list)
    #     Xraysplit = sklearn.model_selection.train_test_split(Xraynoncovid_img_list + Xraycovid_img_list, Xraylabel,
    #                                                        test_size=0.3, random_state=66)
    #     # "0":CT dataset; "1":Xray dataset
    #     ds_train_label = [0] * len(CTsplit[0]) + [1] * len(Xraysplit[0])
    #     ds_val_label = [0] * len(CTsplit[1]) + [1] * len(Xraysplit[1])
    #     dataset_label = {"train":ds_train_label,"val":ds_val_label }
    #     data_split = []
    #     for ct,xray in zip(CTsplit,Xraysplit):
    #         data_split.append(ct+xray)
    #
    #     return data_split,dataset_label

    def init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def train_dataloader(self):
        # transforms
        train_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(90),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=(-5, 5, -5, 5)),
            # random brightness and random contrast
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                 std=[0.33165374, 0.33165374, 0.33165374])
        ])
        # data
        trainset = MixedDataset(self.dataset_info,train=True, transform=train_transformer)
        dataloader = DataLoader(trainset, batch_size=self.hparams.batch_size, drop_last=False, shuffle=True,
                                num_workers=6)

        return dataloader

    def val_dataloader(self):
        val_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                 std=[0.33165374, 0.33165374, 0.33165374])
        ])
        valset = MixedDataset(self.dataset_info, train=False, transform=val_transformer)
        return DataLoader(valset, batch_size=self.hparams.batch_size, drop_last=False, shuffle=False, num_workers=6)

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = StepLR(optimizer, step_size=9999999999999)
        # scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs, self.hparams.cos_lr_min)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        data, label, ds_label = batch
        output = self(data)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., self.hparams.loss_w1]).cuda())
        loss = criterion(output, label.long())
        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y, ds_label = batch
        logits = self(x)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., self.hparams.loss_w1]).cuda())
        loss = criterion(logits, y)
        # compute confucsion matrix on this batch
        pred = logits.argmax(dim=1).view_as(y)
        return {'val_loss': loss, "pred": pred, "label": y, "ds_label":ds_label}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        pred_total = torch.cat([x['pred'] for x in outputs]).view(-1)
        y_total = torch.cat([x['label'] for x in outputs]).view(-1)
        ds_label = torch.cat([x['ds_label'] for x in outputs]).view(-1)

        val_acc = torch.mean((pred_total.cpu() == y_total.cpu()).type(torch.float))
        F1_score = torch.tensor(f1_score(y_total.cpu(),pred_total.cpu(),average="micro"))

        ct_ind = torch.where(ds_label==0)
        CT_F1_score = torch.tensor(f1_score(y_total[ct_ind].cpu(),pred_total[ct_ind].cpu(),average="micro"))
        ct_Confusion_matrix = confusion_matrix(y_total[ct_ind].cpu(), pred_total[ct_ind].cpu())

        xray_ind = torch.where(ds_label==1)
        Xray_F1_score = torch.tensor(f1_score(y_total[xray_ind].cpu(),pred_total[xray_ind].cpu(),average="micro"))
        xray_Confusion_matrix = confusion_matrix(y_total[xray_ind].cpu(), pred_total[xray_ind].cpu())


        print("val_loss = ", avg_loss.cpu())
        print("CT_F1_score=",CT_F1_score)
        print("Xray_F1_score = ",Xray_F1_score)
        print("ct_Confusion_matrix = \n",ct_Confusion_matrix)
        print("xray_Confusion_matrix = \n",xray_Confusion_matrix)
        logs = {"F1_score":F1_score,"val_acc": val_acc,"val_loss":avg_loss,
                "CT_F1_score":CT_F1_score, "Xray_F1_score":Xray_F1_score}
        return {'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        # compute confucsion matrix on this batch
        pred = logits.argmax(dim=1).view_as(y)
        return {'test_loss': loss, "pred": pred, "label": y}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        pred_total = torch.cat([x['pred'] for x in outputs]).view(-1)
        y_total = torch.cat([x['label'] for x in outputs]).view(-1)
        F1_score = f1_score(y_total.cpu(), pred_total.cpu(), average="binary")
        tensorboard_logs = {'test_loss': avg_loss, "F1_score": F1_score}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def on_epoch_start(self):
        if self.hparams.freeze_epochs > 0:
            if self.current_epoch == self.hparams.freeze_epochs:
                self.unfreeze_for_transfer()

    def on_epoch_end(self):
        if self.hparams.log_histogram:
            self.log_histogram()

    def on_train_start(self):
        if self.hparams.freeze_epochs > 0:
            self.freeze_for_transfer()

    """=============================self-defined function============================="""

    def log_histogram(self):
        print("\nlog hist of weights")

        enc_dict = self.model.features.state_dict()
        for name, val in enc_dict.items():
            self.logger.experiment.add_histogram("features/" + name, val, self.current_epoch)

        cls_dict = self.model.classifier.state_dict()
        for name, val in cls_dict.items():
            self.logger.experiment.add_histogram("classifier/" + name, val, self.current_epoch)

    def freeze_for_transfer(self):
        print("Freeze encoder for {} epochs".format(self.hparams.freeze_epochs))
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_for_transfer(self):
        print("\n UnFreeze encoder at {}-th epoch".format(self.hparams.freeze_epochs))
        for param in self.model.features.parameters():
            param.requires_grad = True

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        return parser


def main(args):
    # pick model according to args
    if args.early_stop_callback:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=30,
            strict=True,
            verbose=False,
            mode='min'
        )
    else:
        early_stop_callback = False

    checkpoint_callback = ModelCheckpoint(
        filepath=None,
        monitor='F1_score',
        save_top_k=1,
        mode='max'
    )

    lr_logger = LearningRateLogger()

    if args.test:
        pretrained_model = Mixed_COVID_CT_Xray_Sys.load_from_checkpoint(args.model_path)
        trainer = Trainer(gpus=args.gpus)
        trainer.test(pretrained_model)
        return 0
        # pretrained_model.freeze()
        # y_hat = pretrained_model(x)

    Sys = Mixed_COVID_CT_Xray_Sys(hparams=args)
    trainer = Trainer(early_stop_callback=early_stop_callback,
                      checkpoint_callback=checkpoint_callback,
                      callbacks=[lr_logger],
                      gpus=args.gpus,
                      default_root_dir = '../../results/logs/{}'.format(os.path.basename(__file__)[:-3]),
                      max_epochs=args.max_epochs)

    trainer.fit(Sys)


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument('--dataset_name', type=str, default='Mixed_COVID_CT_Xray', help='')
    parser.add_argument('--dataset_info', type=str, default='dataset_info.json', help='path to datainfo .json file')
    # parser.add_argument('--pretrained_path', type=str, default='DenseNet169/Self-Trans/Self-Trans.pt', help='path to pretrained model')

    parser.add_argument('--early_stop_callback', type=bool, default=False, help='')
    parser.add_argument('--gpus', type=int, default=1, help='')
    parser.add_argument('--test', type=bool, default=False, help='')
    parser.add_argument('--model_path', type=str, default="", help='the well-trained model path for testing')
    parser.add_argument('--freeze_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--cos_lr_min', type=float, default=5e-7)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--loss_w1', type=float, default=0.45,
                        help='CrossEntropy loss weight for COVID type (Majority)')
    parser.add_argument('--num_class', type=int, default=2)
    # Debug Info
    parser.add_argument('--log_histogram', type=bool, default=False, help='')
    parser.add_argument('--note', type=str, default="", help='')
    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args = parser.parse_known_args()

    # let the model add what it wants

    parser = Mixed_COVID_CT_Xray_Sys.add_model_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)
