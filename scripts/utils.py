from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from glob import glob
import torch
import re
import os
import random
import shutil
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from glob import glob
from sklearn import preprocessing
from tqdm import tqdm

def split_image(path, ridge_line):
    """
    Split an image at the ridge line.

    Parameters
    ----------
    path : str
        A path to the image.
    ridge_line : int
        The y axis coordinates of the ridge line.
    
    Returns
    -------
    sky : np.array
        A splitted image of the sky area.
    mnt : np.array
        Another image of the terrain area.
    """
    img = Image.open(path)
    sky = img.crop((0,0,img.size[0],ridge_line))
    mnt = img.crop((0,ridge_line,img.size[0],img.size[1]))
    return sky, mnt

def tile_image(img, num_split):
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
    return croped, bboxes

def draw_box(path, num_split=(6,6), ):
    img = Image.open(path)
    _, bboxes = tile_image(img, num_split)
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
        
def draw_box_ridge(path, ridge_upper, ridge_lower, num_split_sky=(6,3), num_split_ground=(6,3)):
    img = Image.open(path)
    img_sky = img.crop([0, 0, img.size[0], ridge_lower])
    img_ground = img.crop([0, ridge_upper, img.size[0], img.size[1]])
    _, bboxes_sky = tile_image(img_sky, num_split_sky)
    _, bboxes_ground = tile_image(img_ground, num_split_ground)
    p = Path(path)
    d_sky = path.replace(p.suffix, "").\
        replace("sampled", "train_sky")
    d_ground = path.replace(p.suffix, "").\
        replace("sampled", "train_ground")
    
    for name in ["annot_sky", "annot_ground"]:
        d_annot = str(p.parent).replace("sampled", name)
        if os.path.exists(d_annot) == False:
            os.makedirs(d_annot)
        
    for bboxes, name, img_splt, train_d in zip([bboxes_sky, bboxes_ground], ["annot_sky", "annot_ground"], [img_sky, img_ground], [d_sky, d_ground]):
        for i, bbox in enumerate(bboxes):
            img_temp = img_splt.copy()
            draw = ImageDraw.Draw(img_temp)
            draw.rectangle(bbox, outline=(255,0,0), width=4) # 赤枠
            save_path = re.sub("sampled", name, path).\
                replace(".jpg", "_"+str(i).zfill(2)+".jpg")
            img_temp.save(save_path)
            croped = img_splt.crop(bbox)
            train_d2 = train_d + "_" + str(i).zfill(2)
            if os.path.exists(train_d2) == False:
                os.makedirs(train_d2)
            croped.save(train_d2 + "/patch.jpg")
            # img_splt.save(train_d2 + "/all.jpg")
            img.save(train_d2 + "/all.jpg")

def sample_images(in_dir, out_dir, number):
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
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

class PatchedImageDataset(Dataset):
    def __init__(self, annot_path, label_path, image_dir, img_shape=(380, 380)):
        self.annot = pd.read_csv(annot_path)
        self.image_dir = image_dir
        image_dirs = sorted(glob(self.image_dir+"/*"))
        self.image_dirs = []
        for d in image_dirs:
            t = int(d[-7:-5]) # 時間
            if (t > 6) and (t < 17):
                self.image_dirs.append(d)
        self.len = len(self.image_dirs)
        with open(label_path, 'r') as f:
            labels = f.readlines()
            labels = [l.replace("\n", "") for l in labels]
            labels = [int(l) for l in labels]
        self.classes = labels
        self.targets = []
        for d in self.image_dirs:
            image_name = Path(d).stem
            a = self.annot[self.annot['filename'].str.contains(image_name)]["label"].values
            self.targets.append(torch.Tensor(a).int())
        self.targets = torch.concatenate(self.targets)
        self.tf = transforms.Compose(
                [
                    transforms.Resize(img_shape),
                    transforms.ToTensor()#,
                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
    
    def __getitem__(self, index):
        img_dir = self.image_dirs[index]
        img_name = Path(self.image_dirs[index]).stem
        a = self.annot[self.annot['filename'].str.contains(img_name)]["label"]
        a = torch.tensor(a.values).to(torch.long)
        #a = torch.tensor(self.le.transform(a.values)).to(torch.long)
        img_patch = self.tf(Image.open(img_dir + "/patch.jpg"))
        img_all = self.tf(Image.open(img_dir + "/all.jpg"))
        
        return img_patch, img_all, a
    
    def __len__(self):
        return self.len