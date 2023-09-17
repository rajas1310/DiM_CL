import torch
import glob
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

task_dict ={ 'cifar10' : {0 : ['airplane', 'automobile'],
                        1 : ['bird', 'cat'],
                        2 : ['deer', 'dog'],
                        3 : ['frog', 'horse'],
                        4 : ['ship', 'truck']
                        },
             
             'mnist' : { 0 : ['0', '8'],
                        1 : ['1', '7'],
                        2 : ['2', '5'],
                        3 : ['3', '6'],
                        4 : ['4', '9']
                        },
             
             'svhn' : { 0 : ['0', '8'],
                        1 : ['1', '7'],
                        2 : ['2', '5'],
                        3 : ['3', '6'],
                        4 : ['4', '9']
                        }
           }
class ImageDataset(Dataset):
    def __init__(self, args, image_list, label_list, tag='train'):
        super().__init__()
        self.image_list = image_list
        self.label_list = label_list
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(128, antialias=True)])
        if tag == 'train':
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            if args.data == 'mnist':
                self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

        elif tag == 'test':
            if args.data == 'cifar10':
                # for cifar10
                self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))])
            elif args.data == 'mnist':
                self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.131,), (0.308,))])
            elif args.data == 'svhn':
                self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197))])
                
    def __len__(self):
       return len(self.label_list)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.image_list[idx]))
        label = self.label_list[idx]
        return img, label

class DiM_CL_Dataset():
    def __init__(self, args, tasknum, data_dir, tag='train', task_dict=task_dict):
        self.args = args
        self.tag = tag
        if self.tag not in ['train', 'test']:
            self.tag = 'train'
        self.task_dict = task_dict[args.data] # load task dict of a dataset 
        self.task_num = tasknum
        self.data_dir = data_dir
        self.data_images, self.data_labels = [],[]
        self.label2int = self.get_label2int(self.task_dict)
        
        if isinstance(tasknum, int):
            self.get_lists()
        elif isinstance(tasknum, list):
            self.get_alltask_lists()

    def get_label2int(self, task_dict):
        label2int = []
        for task in task_dict.values():
            for clas in task:
                label2int.append(clas)

        keys = [x for x in range(len(label2int))]
        label2int = dict(zip(label2int, keys))
        return label2int

    def  get_lists(self):
        classes = self.task_dict[self.task_num]
        for clas in classes:
            clas_images = glob.glob(f"{self.data_dir}/{self.tag}/{clas}/*.png")
            self.data_images += clas_images
            self.data_labels += [self.label2int[clas] for i in range(len(clas_images))]

    def  get_alltask_lists(self):
        for tnum in self.task_num:
            classes = self.task_dict[tnum]
            for clas in classes:
                clas_images = glob.glob(f"{self.data_dir}/{self.tag}/{clas}/*.png")
                self.data_images += clas_images
                self.data_labels += [self.label2int[clas] for i in range(len(clas_images))]

    def get_dataset(self):
        print(f"INFO : Loaded {(self.tag).upper()} data for TASK {self.task_num}")
        return ImageDataset(self.args, self.data_images, self.data_labels, tag=self.tag)
