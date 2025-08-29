"""
Author: Honggu Liu

"""
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import os
import random
from dataset.transform import xception_default_data_transforms
import cv2
import glob
import numpy as np
class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None,train = False):
        imgs = [] 
        fh = open(txt_path, 'r')
        for line in fh:
            line = line.rstrip()
            words = line.split(',')            
            image_file = words[1]
            path = os.path.join(image_file,'*.png')        
            image_filenames = sorted(glob.glob(path))        
            for i in image_filenames:
                imgs.append((i, int(words[0])))

        print("the number of images: ",len(imgs))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        image = cv2.imread(fn)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations (if any)
        if self.train:
            transform = xception_default_data_transforms['train']
        else:
            transform = xception_default_data_transforms['val']
        
        if self.transform:
            # Apply the selected transform
            image = transform(image=image)["image"]
        
        return image, label
    def __len__(self):
        return len(self.imgs)

