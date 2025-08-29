"""
Author: Andreas Rössler
"""
import albumentations as alb
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from torchvision import transforms

xception_default_data_transforms = {
    'base':
            alb.Compose([
                alb.Resize(299, 299, interpolation=1),
                # alb.Flip(),
                # alb.RandomRotate90(p=0.5),
                # alb.RandomResizedCrop(299, 299, scale=(0.3, 1.0), ratio=(1.0, 1.0), p=0.4), #
                # alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), #  
                # alb.CenterCrop(299, 299),
                alb.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]),
    'train':
            alb.Compose([
                alb.Resize(299, 299, interpolation=1),
                alb.Flip(),
                alb.RandomRotate90(p=0.5),
                alb.RandomResizedCrop(299, 299, scale=(0.3, 1.0), ratio=(1.0, 1.0), p=0.4), #
                alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), #  
                alb.CenterCrop(299, 299),
                alb.ToGray(p=0.1),
                #alb.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                #alb.GaussianBlur(blur_limit=3, sigma_limit=0, p=0.05),
                #alb.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.1),
                # alb.OneOf([
                #     alb.RandomBrightnessContrast(),
                #     alb.FancyPCA(),
                #     alb.HueSaturationValue(),
                # ]),
                alb.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]),
    'val':
            alb.Compose([
                alb.Resize(299, 299, interpolation=1),
                alb.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),
                ToTensorV2(),

            ]),
    'test': 
            alb.Compose([
                alb.Resize(299, 299, interpolation=1),
                alb.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]),
}

xception_default_data_transforms_256 = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Resize((256, 256)),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}
transforms_224 = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}