import torch
import glob
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

task_dict = {0 : ['airplane', 'automobile'],
         1 : ['bird', 'cat'],
         2 : ['deer', 'dog'],
         3 : ['frog', 'horse'],
         4 : ['ship', 'truck']}

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
    def __init__(self, tasknum, data_dir, tag='train', task_dict=task_dict):
        self.tag = tag
        if self.tag not in ['train', 'test']:
            self.tag = 'train'
        self.task_dict = task_dict
        self.task_num = tasknum
        self.data_dir = data_dir
        self.data_images, self.data_labels = [],[]
        self.label2int = { 'airplane' : 0, 'automobile' : 1,
         'bird' : 2, 'cat' : 3,
         'deer' : 4, 'dog' : 5,
         'frog' : 6, 'horse' : 7,
         'ship' : 8, 'truck' : 9}

        self.get_lists()
        
    def  get_lists(self):
        classes = self.task_dict[self.task_num]
        for clas in classes:
            clas_images = glob.glob(f"{self.data_dir}/{self.tag}/{clas}/*.png")
            self.data_images += clas_images
            self.data_labels += [self.label2int[clas] for i in range(len(clas_images))]

    def get_dataset(self):
        print(f"INFO : Loaded {(self.tag).upper()} data for TASK {self.task_num}")
        return ImageDataset(self.data_images, self.data_labels, tag=self.tag)