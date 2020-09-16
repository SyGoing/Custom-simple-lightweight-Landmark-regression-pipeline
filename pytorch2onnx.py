import torch
import torch.nn as nn
import numpy as np
from torchvision import  transforms
from networks.Onet import ONet

def export_onnx(input_path,output_path):
    net=ONet(4)
    net.load_state_dict(torch.load(input_path))
    net.eval()

    input=torch.randn(1,3,48,120)
    out=net(input)

    torch.onnx.export(net,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      output_path,  # where to save the model (can be a file or file-like object)
                      verbose=True,
                      )

def parse_args():
    parser = argparse.ArgumentParser(description='Trainging Template')

    parser.add_argument('--i', default="./models/demo.pth", type=str,help="input model path")
    parser.add_argument('--o', default="./onnxmodel/demo.onnx", type=str, help="output onnx  path")

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args()
    export_onnx(args.i,args.o)



