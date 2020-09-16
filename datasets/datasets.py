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

HEIGHT=48
WWIDTH=120

#一个好的想法就是用labelme标记好数据以后做多任务包括关键点回归的训练
#labelme
# if the data is marked by the json after being croped by bbox
class CustomJsonDatasets(data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.landmarks = None
        self.transforms = transforms
        self.data_list=self.load_data_json(data_dir)

    def __getitem__(self, index):
        img_path, self.landmarks=self.data_list[index]
        self.img = Image.open(img_path).convert('RGB')
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.landmarks)

    def __len__(self):
        return len(self.data_list)

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
        for x in glob.glob(data_dir + '/images/*.jpg', recursive=True):
            d = pathlib.Path(x)
            label_path = os.path.join(data_dir, 'annotations', (str(d.stem) + '.json'))
            landmarks= self._get_annotation_json(label_path)
            if len(landmarks) > 0:
                data_list.append((x, landmarks))
            else:
                print('there is no suit bbox on {}'.format(label_path))
        return data_list


class WLFWDatasets(data.Dataset):
    def __init__(self, file_list,kpnum, transforms=None):
        self.line = None
        self.path = None
        self.kpnum=kpnum
        self.landmarks = None
        self.attribute = None
        self.filenames = None
        self.euler_angle = None
        self.transforms = transforms
        self.imgs_root = os.path.dirname(file_list)
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split(' ')
        self.img = Image.open(os.path.join(self.imgs_root, self.line[0])).convert('RGB')
        #image=cv2.imread(os.path.join(self.imgs_root, self.line[0]))
        #self.img=Image.fromarray(image)


        self.landmark = np.asarray(self.line[1:self.kpnum*2+1], dtype=np.float32)
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.landmark)

    def __len__(self):
        return len(self.lines)


if __name__ == '__main__':
    list="../data/processed_data/landmark_list.txt"
    datset=WLFWDatasets(list,4,transforms=transforms.ToTensor())
    dataloader=DataLoader(datset,batch_size=1, shuffle=False, num_workers=1)

    for img, landmark in dataloader:
        print("img shape", img.shape)
        print("landmark size", landmark.size())
    print(len(datset))