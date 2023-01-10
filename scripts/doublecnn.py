import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader, Subset
from scripts.utils import PatchedImageDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

from dataclasses import dataclass
import yaml
from tqdm import tqdm
import datetime

from PIL import Image, ImageDraw, ImageFont
from scripts.utils import tile_image
from glob import glob
import pandas as pd
import os
import shutil

@dataclass(init=True)
class Config:
    # General information for model training
    annot_path:  str
    label_path: str
    image_dir: str
    model_name: str
    val_ratio: float
    batch_size: int
    num_workers: int
    device: str
    region: str
    ridge: int
    num_split: list
    # Hyperparameters of the model
    n_layers: int
    h_dims: int
    batch_norm: bool
    drop_out: float
    learning_rate: float
    freeze_base: bool

class DoubleCNN(nn.Module):
    def __init__(self, model_name, y_dims, n_layers, h_dims, batch_norm, drop_out, freeze = False):
        super().__init__()
        self.modelp = getattr(models, model_name)(pretrained=True)
        self.modela = getattr(models, model_name)(pretrained=True)
        if freeze:
            for model in [self.modela, self.modelp]:
                for param in model.parameters():
                    param.requires_grad = False
        if "resnet" in model_name:
            self.modelp.fc = nn.Linear(self.modelp.fc.in_features, h_dims)
            self.modela.fc = nn.Linear(self.modela.fc.in_features, h_dims)
        elif "efficientnet" in model_name:
            self.modelp.classifier[1] = nn.Linear(self.modelp.classifier[1].in_features, h_dims)
            self.modela.classifier[1] = nn.Linear(self.modela.classifier[1].in_features, h_dims)
        if n_layers == 1:
            self.fc = nn.Linear(h_dims, y_dims)
        else:
            self.fc = [nn.Linear(h_dims, h_dims)]
            for i in range(n_layers-1):
                self.fc.append(nn.Linear(h_dims, h_dims))
                self.fc.append(nn.GELU())
                if batch_norm:
                    self.fc.append(nn.BatchNorm1d(h_dims))
                self.fc.append(nn.Dropout(drop_out))
            self.fc.append(nn.Linear(h_dims, y_dims))
            self.fc = nn.Sequential(*self.fc)
    
    def forward(self, x_patch, x_all):
        hp = self.modelp(x_patch)
        ha = self.modela(x_all)
        #h = torch.cat([hp, ha], 1)
        h = hp + ha
        return self.fc(h)

class DoubleCNNClassifier:
    def __init__(self, config_path):
        # Loading configuration file
        self.config_path = config_path
        with open(config_path) as file:
            self.c = Config(**yaml.safe_load(file))
        # Setting input size of each CNN
        if self.c.model_name == "efficientnet_b0":
            self.input_size = (224, 224)
        elif self.c.model_name == "efficientnet_b1":
            self.input_size = (240, 240)
        elif self.c.model_name == "efficientnet_b2":
            self.input_size = (260, 260)
        elif self.c.model_name == "efficientnet_b3":
            self.input_size = (300, 300)
        elif self.c.model_name == "efficientnet_b4":
            self.input_size = (380, 380)
        elif self.c.model_name == "efficientnet_b5":
            self.input_size = (456, 456)
        elif self.c.model_name == "efficientnet_b6":
            self.input_size = (528, 528)
        elif self.c.model_name == "efficientnet_b7":
            self.input_size = (600, 600)
        elif self.c.model_name == "efficientnet_v2_s":
            self.input_size = (384, 384)
        elif self.c.model_name == "efficientnet_v2_m":
            self.input_size = (480, 480)
        elif self.c.model_name == "efficientnet_v2_l":
            self.input_size = (480, 480)
        elif "resnet" in self.c.model_name:
            self.input_size = (224, 224)
        else:
            raise ValueError("Invalid model name!")
        
        # Constructing Dataloaders
        self.ds = PatchedImageDataset(self.c.annot_path, self.c.label_path, self.c.image_dir, self.input_size)
        self.idx = list(range(len(self.ds.targets)))
        train_index, val_index = train_test_split(range(len(self.ds)), test_size=self.c.val_ratio, stratify=self.ds.targets, shuffle=True)
        self.train_loader = DataLoader(Subset(self.ds, train_index), batch_size=self.c.batch_size, \
            num_workers=self.c.num_workers, shuffle=True)
        self.val_loader = DataLoader(Subset(self.ds, val_index), batch_size=self.c.batch_size, \
            num_workers=self.c.num_workers, shuffle=False)
        self.classes = self.ds.classes
        # Constructing models
        self.model = DoubleCNN(self.c.model_name, len(self.classes), self.c.n_layers, \
            self.c.h_dims, self.c.batch_norm, self.c.drop_out, self.c.freeze_base)
        self.model.to(self.c.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.c.learning_rate)
    
    def _train(self, epoch):
        self.model.train()
        running_loss = 0.0
        for x_patch, x_all, y in tqdm(self.train_loader):
            x_patch = x_patch.to(self.c.device)
            x_all = x_all.to(self.c.device)
            y = y.to(self.c.device).to(torch.long).squeeze()
            
            self.optimizer.zero_grad()
            y2 = self.model(x_patch, x_all)
            loss = self.criterion(y2, y)
            running_loss += loss

            loss.backward()
            self.optimizer.step()
        running_loss = running_loss / len(self.train_loader.dataset)
        print("Epoch {} train_loss: {:.4f}".format(epoch, running_loss))
        return running_loss
    
    def _val(self, epoch):
        self.model.eval()
        running_loss = 0.0
        ys = []
        pred_ys = []
        for x_patch, x_all, y in tqdm(self.val_loader):
            x_patch = x_patch.to(self.c.device)
            x_all = x_all.to(self.c.device)
            y = y.to(self.c.device).to(torch.long).squeeze()
            
            with torch.no_grad():
                y2 = self.model(x_patch, x_all)
            loss = self.criterion(y2, y)
            running_loss += loss
            _, pred = torch.max(y2, 1)
            ys.append(y.detach().cpu())
            pred_ys.append(pred.detach().cpu())
            
        running_loss = running_loss / len(self.val_loader.dataset)
        ys = torch.cat(ys)
        pred_ys = torch.cat(pred_ys)
        res = classification_report(ys, pred_ys, output_dict=True)
        print("Epoch {} val_loss: {:.4f}".format(epoch, running_loss))
        print("Epoch {} val_f1: {:.4f}".format(epoch, res["macro avg"]["f1-score"]))
        return running_loss, res
    
    def train(self, epochs, fold = None):
        try:
            self.out_dir
        except AttributeError:
            dt_now = datetime.datetime.now()
            exp_time = dt_now.strftime('%Y%m%d_%H_%M_%S')
            self.out_dir = "./runs/" + self.c.model_name + "_" + self.c.region + "_" + exp_time
        if fold is None:
            writer = SummaryWriter(self.out_dir)
        else:
            writer = SummaryWriter(self.out_dir + "/fold_" + str(fold))
        shutil.copy(self.config_path, self.out_dir + "/" + "config.yaml")
        self.best_loss = 9999
        for epoch in range(epochs):
            train_loss = self._train(epoch)
            val_loss, res = self._val(epoch)
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("validation_loss", val_loss, epoch)
            for c, r in res.items():
                if c != "accuracy":
                    r = {k: v for k, v in r.items() if k.lower() != "support"}
                    writer.add_scalars("val_" + c, r, epoch)
            if val_loss <= self.best_loss:
                self.save(self.out_dir + "/" + "best.pth")
                self.best_epoch = epoch
                self.best_loss = val_loss
                self.best_f1 = res["macro avg"]["f1-score"]
                self.best_metrics = res
        writer.close()
    
    def kfold(self, epochs, k=5):
        kf = StratifiedKFold(n_splits=k, shuffle=True)
        results = []
        for fold, (train_indices, val_indices) in enumerate(kf.split(self.idx, self.ds.targets)):
            # Reset the dataloaders and the model
            train_dataset = torch.utils.data.Subset(self.ds, train_indices)
            val_dataset = torch.utils.data.Subset(self.ds, val_indices)
            self.train_loader = DataLoader(train_dataset, batch_size=self.c.batch_size, \
                num_workers=self.c.num_workers, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=self.c.batch_size, \
                num_workers=self.c.num_workers, shuffle=False)
            self.model = DoubleCNN(self.c.model_name, len(self.classes), self.c.n_layers, \
                self.c.h_dims, self.c.batch_norm, self.c.drop_out).to(self.c.device)
            self.optimizer = optim.RAdam(self.model.parameters(), lr=self.c.learning_rate)
            # train
            self.train(epochs, fold)
            df = pd.DataFrame(self.best_metrics)
            df["metrics"] = df.index
            df = pd.melt(df, id_vars="metrics", var_name="class", value_name="value")
            df["fold"] = fold
            results.append(df)
        pd.concat(results).to_csv(self.out_dir + "/staritfied_cv.csv")
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def predict(self, path):
        img = Image.open(path)
        if self.c.region == "sky":
            img_croped = img.crop([0, 0, img.size[0], self.c.ridge])
        elif self.c.region == "ground":
            img_croped = img.crop([0, self.c.ridge, img.size[0], img.size[1]])
        _, bboxes = tile_image(img_croped, self.c.num_split)
        x_patch = torch.stack([self.ds.tf(img_croped.crop(bbox)) for bbox in bboxes]).to(self.c.device)
        x_all = self.ds.tf(img).repeat(len(bboxes), 1, 1, 1).to(self.c.device)
        self.model.eval()
        with torch.no_grad():
            y2 = self.model(x_patch, x_all)
            _, pred = torch.max(y2, 1)
        pred = pred.detach().cpu()
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 192)
        colors = {
            "0": (68,1,84,0),
            "1": (49,104,142,0),
            "2": (53,183,121,0),
            "error": (255,0,0,0)
        }
        i = 0
        datetime = path[-17:-4]
        datetimes = []
        classes = []
        index = []
        for bbox, p in zip(bboxes, pred):
            if self.c.region == "ground":
                bbox[1] += self.c.ridge
                bbox[3] += self.c.ridge
            draw.rectangle(bbox, outline=(255,0,0), width=3)
            if p == 3:
                p = "error"
            else:
                p = str(p.item())
            draw.text(
                (bbox[0]+5,bbox[1]+5), 
                p, 
                colors[p], 
                font=font,
                stroke_width=8,
                stroke_fill="white")
            index.append(i)
            classes.append(p)
            datetimes.append(datetime)
            i+=1
        df = pd.DataFrame({
            "datetime": datetimes,
            "index": index,
            "weather": classes
        })
        return df, img
    
    def predict_dir(self, d, rename_from, rename_to, use_best=True):
        out = d.replace(rename_from, rename_to)
        if os.path.exists(out) == False:
            os.makedirs(out)
        dfs = []
        if use_best:
            self.load(self.out_dir + "/" + "best.pth")
        for path in tqdm(sorted(glob(d + "/*"))):
            out_path = path.replace(rename_from, rename_to)
            df, img = self.predict(path)
            img.save(out_path)
            dfs.append(df)
        df = pd.concat(dfs)
        return df
    
    def load(self, path):
        if self.c.device == "cpu":
            self.model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        else:
            self.model.load_state_dict(torch.load(path))
    
    def test(self, image_dir, annot_path, result_yaml_path=None):
        self.test_ds = PatchedImageDataset(annot_path, self.c.label_path, image_dir, self.input_size)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.c.batch_size, \
                num_workers=1, shuffle=False)
        self.model.eval()
        running_loss = 0.0
        ys = []
        pred_ys = []
        for x_patch, x_all, y in tqdm(self.test_loader):
            x_patch = x_patch.to(self.c.device)
            x_all = x_all.to(self.c.device)
            y = y.to(self.c.device).to(torch.long).squeeze()
            
            with torch.no_grad():
                y2 = self.model(x_patch, x_all)
            loss = self.criterion(y2, y)
            running_loss += loss
            _, pred = torch.max(y2, 1)
            ys.append(y.detach().cpu())
            pred_ys.append(pred.detach().cpu())
            
        running_loss = running_loss / len(self.test_loader.dataset)
        ys = torch.cat(ys)
        pred_ys = torch.cat(pred_ys)
        res = classification_report(ys, pred_ys, output_dict=True)
        print("Val_loss: {:.4f}".format(running_loss))
        print("Val_f1: {:.4f}".format(res["macro avg"]["f1-score"]))
        if result_yaml_path is not None:
            with open(result_yaml_path, "w") as f:
                yaml.dump(res, f)
        return running_loss, res