from __future__ import print_function, division
from __future__ import absolute_import

import argparse
from networks.Onet import ONet
from loss import loss
from datasets import KPTDatasets

import os
import cv2
import numpy as np

import torch
from torchvision import transforms
from tensorboardX import SummaryWriter

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        print(s)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print('Save checkpoint to {0:}'.format(filename))

def adjust_learning_rate(optimizer, initial_lr, step_index):
    lr = initial_lr * (0.1 ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_warmup(optimizer, epoch,base_learning_rate,batch_id,burn_in=1000):
    lr = base_learning_rate
    if batch_id<burn_in:
        lr=base_learning_rate*pow(float(batch_id)/float(burn_in),4)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if epoch >=60:  #40 320
            lr /= 10
        if epoch >=120 :
            lr /= 10
        if epoch >= 267:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(net,trainloader,criterion,optimizer,epoch):
    train_loss = 0
    batch_num=0
    for imgs, landmark_gt in trainloader:
        batch_num+=1
        imgs, landmark_gt = imgs.cuda(), landmark_gt.cuda()
        optimizer.zero_grad()

        landmark_pred=net(imgs)
        lds_loss=criterion(landmark_pred,landmark_gt)
        lds_loss.backward()
        optimizer.step()

        train_loss+=lds_loss.data.item()

    return train_loss/batch_num


def validate(val_dataloader, net, args):
    net.eval()

    with torch.no_grad():
        losses_NME = []
        losses_ION = []
        pose_losses_MAE = []
        for img, landmark_gt  in val_dataloader:
            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            landmark_gt.requires_grad = False
            landmark_gt = landmark_gt.cuda(non_blocking=True)

            net = net.cuda()
            landmarks = net(img)
            landmarks = landmarks[0]
            landmark_gt = landmark_gt[0]
            if landmarks.shape[0] > 0:
                loss = torch.mean(
                    torch.sqrt(torch.sum((landmark_gt * 48 - landmarks  * 48)**2, dim=1))
                    )

                landmarks = landmarks.cpu().numpy()
                landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
                landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()


                error_diff = np.sum(np.sqrt(np.sum((landmark_gt - landmarks) ** 2, axis=2)), axis=1)
                interocular_distance = np.sqrt(np.sum((landmarks[:, 60, :] - landmarks[:,72, :]) ** 2, axis=1))
                error_norm = np.mean(error_diff / interocular_distance)
                losses_NME.append(loss.cpu().numpy())
                losses_ION.append(error_norm)
        return np.mean(losses_NME), np.mean(losses_ION)


def main(args):
    print_args(args)

    #Net
    net =ONet(4)
    net.cuda()

    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))
        print("load %s successfully ! " % args.resume)
    print(net)


    #super params
    step_epoch = [int(x) for x in args.step.split(',')]

    criterion = WingLoss()
    cur_lr = args.base_lr
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay)

    #dataset
    train_transform = transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(0.5, 0.5, 0.5)),
    ])

    train_datasets=KPTDatasets(args.dataroot, train_transform)
    dataloader = DataLoader(
        train_datasets,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    step_index = 0
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_lds_loss = train(net,dataloader,
                                      criterion, optimizer, epoch)
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth')
        save_checkpoint(net.state_dict(),filename)
        if epoch in step_epoch:
            step_index += 1
            cur_lr = adjust_learning_rate(optimizer, args.base_lr, step_index)

        print('Epoch: %d,  train lds loss:%6.4f, lr:%8.6f'%(epoch, train_lds_loss, cur_lr))
        writer.add_scalar('data/train_lds_loss', train_lds_loss, epoch)
    writer.close()



def parse_args():
    parser = argparse.ArgumentParser(description='Trainging Template')
    # general
    parser.add_argument('-j', '--workers', default=8, type=int)

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
    parser.add_argument('--step', default="30,60", help="lr decay", type=str)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=80, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',default='./models/checkpoint/snapshot/',type=str,metavar='PATH')
    parser.add_argument('--tensorboard', default="./models/checkpoint/tensorboard", type=str)
    parser.add_argument('--resume', default='./models/checkpoint/model-wing/checkpoint_epoch_5.pth', type=str, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument('--dataroot',default='./data',type=str,metavar='PATH')
    parser.add_argument('--train_batchsize', default=128, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)