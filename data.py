import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import transforms

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),   # 随机将图片水平翻转
    transforms.RandomRotation(15),   # 随机旋转图片
    transforms.ToTensor(),    # 将图片类型转化为tensor 并将数值normalize 到[0， 1]
])

validation_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class MyData(Dataset):
    def __init__(self, root_dir, flag):
        self.root_dir = root_dir
        self.image_path = os.listdir(root_dir)
        self.flag = flag
        if self.flag == 0:
            self.transform = train_transform
        else:
            self.transform = validation_transform

    def __getitem__(self, idx):
        image_name = self.image_path[idx]
        image_item_path = os.path.join(self.root_dir, image_name)
        img = Image.open(image_item_path)
        img = self.transform(img)
        label = torch.LongTensor(eval('[' + image_name.split("_")[0] + ']'))
        return img, label

    def __len__(self):
        return len(self.image_path)


