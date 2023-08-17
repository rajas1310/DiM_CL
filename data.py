import torch
import glob
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

task_dict = {1 : ['airplane', 'automobile'],
         2 : ['bird', 'cat'],
         3 : ['deer', 'dog'],
         4 : ['frog', 'horse'],
         5 : ['ship', 'truck']}

class ImageDataset(Dataset):
    def __init__(self, image_list, label_list, tag='train'):
        super().__init__()
        self.image_list = image_list
        self.label_list = label_list
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(128, antialias=True)])
        if tag == 'train':
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif tag == 'test':
            # for cifar10
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))])

    def __len__(self):
       return len(self.label_list)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.image_list[idx]))
        label = self.label_list[idx]
        return img, label

class DiM_CL_Dataset():
    def __init__(self, tasknum, data_dir, task_dict=task_dict):
        self.task_dict = task_dict
        self.task_num = tasknum
        self.data_dir = data_dir
        self.train_images, self.train_labels = [],[]
        self.test_images, self.test_labels = [],[]
        self.get_lists()
        
    def  get_lists(self):
        classes = self.task_dict[self.task_num]
        for clas in classes:
            clas_images = glob.glob(f"{self.data_dir}/train/{clas}/*.png")
            self.train_images += clas_images
            self.train_labels += [clas for i in range(len(clas_images))]

        for t in range(1,self.task_num):
            classes += self.task_dict[self.task_num]
        for clas in classes:
            clas_images = glob.glob(f"{self.data_dir}/test/{clas}/*.png")
            self.test_images += clas_images
            self.test_labels += [clas for i in range(len(clas_images))]
    
    def get_datasets(self):
        return ImageDataset(self.train_images, self.train_labels), ImageDataset(self.test_images, self.test_labels)