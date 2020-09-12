import numpy as np
import cv2
import sys
from torch.utils import data
from torch.utils.data import DataLoader
import json
import glob
import pathlib
import os
from PIL import Image
from torchvision import transforms

class KPTDatasets(data.Dataset):
    def __init__(self, data_dir, transforms=None):

        self.landmarks = None
        self.transforms = transforms
        self.data_list=self.load_data_json(data_dir)

    def __getitem__(self, index):
        img_path, self.landmark=self.data_list[index]
        self.img = Image.open(img_path).convert('RGB')
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.landmark)

    def __len__(self):
        return len(self.data_list)

    # if the data is marked by the json
    def _get_annotation_json(self,label_path):
        landmarks = []
        data = json.load(open(label_path))
        h=float(data['imageHeight'])
        w=float(data['imageWidth'])


        for shape in data['shapes']:
            points = shape['points']
            for pt in points:
                landmarks.append(pt[0]/w)
                landmarks.append(pt[1]/h)
        return np.array(landmarks, dtype=np.float32)

    def load_data_json(self,data_dir):
        data_list = []
        for x in glob.glob(data_dir + '/imgs/*.jpg', recursive=True):
            d = pathlib.Path(x)
            label_path = os.path.join(data_dir, 'gt', (str(d.stem) + '.json'))
            landmarks= self._get_annotation_json(label_path)
            if len(landmarks) > 0:
                data_list.append((x, landmarks))
            else:
                print('there is no suit bbox on {}'.format(label_path))
        return data_list

if __name__ == '__main__':
    data_dir="./data"
    datset=KPTDatasets(data_dir,transforms=transforms.ToTensor())
    dataloader=DataLoader(datset,batch_size=1, shuffle=False, num_workers=1)

    for img, landmark in dataloader:
        print("img shape", img.shape)
        print("landmark size", landmark.size())
    print(len(datset))