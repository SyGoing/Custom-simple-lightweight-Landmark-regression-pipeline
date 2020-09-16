import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import  transforms
import cv2
from networks.Onet import ONet
import os

import argparse
def infer_image(image_path,model,transform):
    image=cv2.imread(image_path)
    input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input = transform(input).unsqueeze(0).to(device)
    landmarks = model(input)
    pre_landmark = landmarks.cpu().detach().numpy().reshape(-1, 2) * [120, 48]

    for (x,y) in pre_landmark.astype(np.float32):
        cv2.circle(image, ( x,  y), 1, (0, 0, 255))

    cv2.namedWindow("kp",cv2.WINDOW_NORMAL)
    cv2.imshow("kp",image)
    cv2.waitKey()

def inferimgs(image_path,model,transform):
    files=os.listdir(image_path)
    for file in files:
        infer_image(os.path.join(image_path,file),model,transform)




#for another mean and std ,do
def pre_process(img,inp_width,inp_height):
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

    inp_image=cv2.resize(img,(inp_width,inp_height))
    inp_image=((inp_image/255.-mean)/std).astype(np.float32)
    inp = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    return inp

"""
transform = transforms.Compose([transforms.ToTensor()])  ------>the same as follows steps
inp_image=(inp_image/255.).astype(np.float32)
 inp = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
"""

"""
img  (c
image=Image.fromarray(img)



"""

def PILImageToCV(PILImage):
   # PIL Image转换成OpenCV格式
    img = cv2.cvtColor(np.asarray(PILImage), cv2.COLOR_RGB2BGR)
    return img

def CVImageToPIL(CVImage):
    img = Image.fromarray(cv2.cvtColor(CVImage, cv2.COLOR_BGR2RGB))
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='Trainging Template')
    # general
    parser.add_argument('-j', '--workers', default=2, type=int)

    # test
    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--test_type', default="images", type=str,help="image/webcam/video/images")
    parser.add_argument('--data_path', default="./data/processed_data/croped_images", type=str, help="./name.jpg/name.mp4")
    parser.add_argument('--model_path',default='./models/checkpoint/snapshot/checkpoint_epoch_158.pth',type=str,metavar='PATH')
    parser.add_argument('--test_batchsize', default=1, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = ONet(4)
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint)
    net.to(device)
    net.eval()

    transform = transforms.Compose([transforms.ToTensor()])

    if args.test_type=="image":
        infer_image(args.data_path,net,transform)
    elif args.test_type=="images":
        inferimgs(args.data_path,net,transform)
    elif args.test_type=="video":
        pass











