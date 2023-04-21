import os
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
import random

class Dataset(data.Dataset):
    def __init__(self, img_list, transform=None):
        self.transform = transform
        self.img_paths = []
        self.labels = []
        with open(img_list, 'r') as fd:
            lines = fd.readlines()
        random.shuffle(lines)
        for line in lines:
            words = line.strip().split()
            self.labels.append(np.int32(words[1:]))
            self.img_paths.append(words[0])
            # try:
            #     self.labels.append(np.int32(words[1:]))
            #     self.img_paths.append(words[0])
            # except:
            #     print(words[0])

    def __getitem__(self, index):
        #import pdb;pdb.set_trace()
        img_path = self.img_paths[index]
        data = cv2.imread(img_path)
        if data is None:
            print('no img', img_path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        # data augment
        data = self.transform(data)
        label = np.array(self.labels[index])
        label = torch.from_numpy(label)
        #print(data.shape, label.shape)
        return data, ( label[0], label[1],label[2])
        #return data, label

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    train_data = Dataset("../dataset/train.txt", "train")
    trainloader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    for i, (data, label) in enumerate(trainloader):
        img = torchvision.utils.make_grid(data).numpy()
        img = np.transpose(img, (1, 2, 0))
        #img *= np.array([0.229, 0.224, 0.225])
        #img += np.array([0.485, 0.456, 0.406])
        img *= 255
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        cv2.imshow('img', img)
        if cv2.waitKey(10000):
            break
