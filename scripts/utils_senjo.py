from cProfile import label
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from glob import glob
import torch
import re
import os
import random
import shutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from glob import glob
import numpy as np
from sklearn import preprocessing

def split_image(path, num_split):
    img = Image.open(path)
    img = img.crop((0,0,img.size[0],700)) # 925より下は近景なので消す

    height = img.size[1] // (num_split[1])
    width = img.size[0] // (num_split[0])
    bboxes = []
    croped = []
    # 縦の分割枚数
    for h1 in range(num_split[1]):
        # 横の分割枚数
        for w1 in range(num_split[0]):
            w2 = w1 * width
            h2 = h1 * height
            bbox = [w2, h2, w2 + width, h2 + height]
            bboxes.append(bbox)
            croped.append(to_tensor(img.crop(bbox)))
    croped = torch.stack(croped)
    return croped, bboxes, img

def draw_box(path, num_split=(6,6)):
    _, bboxes, img = split_image(path, num_split)
    p = Path(path)
    d = path.replace(p.suffix, "").\
        replace("sampled", "train")
    if os.path.exists(d) == False:
        os.makedirs(d)
    for i, bbox in enumerate(bboxes):
        img_temp = img.copy()
        draw = ImageDraw.Draw(img_temp)
        draw.rectangle(bbox, outline=(255,0,0), width=4) # 赤枠
        save_path = re.sub("sampled", "annot", path).\
            replace(".jpg", "_"+str(i).zfill(2)+".jpg")
        img_temp.save(save_path)
        croped = img.crop(bbox)
        croped.save(d + "/" + str(i).zfill(2) + ".jpg")
        

def sample_images(in_dir, out_dir, number):
    in_imgs = glob(in_dir + "/*")
    sampled = random.sample(in_imgs, number)
    for s in sampled:
        shutil.copy(s, s.replace(in_dir, out_dir))

def filter_images(in_dir, from_time, to_time):
    in_imgs = glob(in_dir + "/*")
    for img in in_imgs:
        t = int(img[-8:-6]) # 6 ~ 17の数字
        if (t < from_time) or (t > to_time):
            os.remove(img)

#filter_images("data/senjo/2021", 7, 16)
sample_images("data/senjo/2019", "data/senjo/sampled2", 50)

for img in sorted(glob("data/senjo/sampled2/*")):
    draw_box(img, (4, 5))

class PatchedImageDataset(Dataset):
    def __init__(self, annot_path, label_path, image_dir):
        self.annot = pd.read_csv(annot_path)
        self.image_dir = image_dir
        self.image_dirs = sorted(glob(self.image_dir+"/*"))
        self.len = len(self.image_dirs)
        with open(label_path, 'r') as f:
            labels = f.readlines()
            labels = [l.replace("\n", "") for l in labels]
        self.le = preprocessing.LabelEncoder()
        self.le.fit_transform(labels)
        self.classes = self.le.classes_
        self.targets = self.annot["label"].values
    
    def __getitem__(self, index):
        img_name = Path(self.image_dirs[index]).stem
        a = self.annot[self.annot['filename'].str.contains(img_name)]["label"]
        a = torch.tensor(self.le.transform(a.values)).to(torch.long)
        img_paths = glob(self.image_dirs[index]+"/*")
        imgs = [Image.open(p) for p in img_paths]
        self.tf = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        imgs = torch.stack([self.tf(img) for img in imgs])
        return imgs, a
    
    def __len__(self):
        return self.len

