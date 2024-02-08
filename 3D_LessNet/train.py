import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms
from Models import *
# from Functions import TrainDataset
import torch.utils.data as Data
from data import datasets, trans
from natsort import natsorted
import csv
parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=320001,
                    help="number of total iterations")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.02,
                    help="labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=403,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=4,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
trainingset = opt.trainingset
using_l2 = opt.using_l2

def dice(pred1, truth1):
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    dice_35=np.zeros(len(VOI_lbls))
    index = 0
    for k in VOI_lbls:
        #print(k)
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        # print(intersection)
        dice_35[index]=intersection / (np.sum(pred) + np.sum(truth))
        index = index + 1
    return np.mean(dice_35)

def save_checkpoint(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    # print(model_lists)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def train():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    atlas_dir = '/bask/projects/d/duanj-ai-imaging/UvT/TransMorph_Xi/IXI_Mine/IXI_data/atlas.pkl'
    train_dir = '/bask/projects/d/duanj-ai-imaging/UvT/TransMorph_Xi/IXI_Mine/IXI_data/Train/'
    val_dir = '/bask/projects/d/duanj-ai-imaging/UvT/TransMorph_Xi/IXI_Mine/IXI_data/Val/'
    # train_composed = transforms.Compose([trans.RandomFlip(0),
                                         # trans.NumpyType((np.float32, np.float32)),
                                         # ])

    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
    val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    train_loader = Data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = Data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)


    model = UNet(6, 3, start_channel).to(device)
    if using_l2 == 1:
        loss_similarity = MSE().loss
    elif using_l2 == 0:
        loss_similarity = SAD().loss
    elif using_l2 == 2:
        loss_similarity = NCC()
    loss_smooth = smoothloss

    transform = SpatialTransform().to(device)
    diff_transform = DiffeomorphicTransform(time_step=7).to(device)
    

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = './L2ss_{}_Chan_{}_Smth_{}_LR_{}_Val/'.format(using_l2,start_channel,smooth,lr)
    csv_name = 'L2ss_{}_Chan_{}_Smth_{}_LR_{}.csv'.format(using_l2,start_channel,smooth,lr)
    assert os.path.exists(csv_name) ==0
    assert os.path.isdir(model_dir) ==0
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((3, iteration))
    step = 1
    epoch = 0
    while step <= iteration:
        for X, Y in train_loader:

            X = X.to(device).float()
            Y = Y.to(device).float()
            
            f_xy = model(X, Y)
            # D_f_xy = diff_transform(f_xy)
            D_f_xy = f_xy
            X_Y = transform(X, D_f_xy.permute(0, 2, 3, 4, 1))
            
            loss1 = loss_similarity(Y, X_Y)
            loss5 = loss_smooth(f_xy)
            loss = loss1 + smooth * loss5
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step] = np.array([loss.item(),loss1.item(),loss5.item()])
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" -smo "{3:.4f}" '.format(step, loss.item(),loss1.item(),loss5.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0) or (step == 1):
                with torch.no_grad():
                    Dices_Validation = []
                    for data in val_loader:
                        model.eval()
                        xv = data[0]
                        yv = data[1]
                        xv_seg = data[2]
                        yv_seg = data[3]
                        vf_xy = model(xv.float().to(device), yv.float().to(device))
                        warped_xv_seg= transform(xv_seg.float().to(device), vf_xy.permute(0, 2, 3, 4, 1), mod = 'nearest')
                        for bs_index in range(bs):
                            dice_bs=dice(warped_xv_seg[bs_index,...].data.cpu().numpy().copy(),yv_seg[bs_index,...].data.cpu().numpy().copy())
                            Dices_Validation.append(dice_bs)
                    modelname = 'DiceVal_{:.4f}_Epoch_{:04d}.pth'.format(np.mean(Dices_Validation), epoch)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, np.mean(Dices_Validation)])
                    save_checkpoint(model.state_dict(), model_dir, modelname)
                modelname = 'Epoch_{:09d}.pth'.format(epoch)
                torch.save(model.state_dict(), model_dir + modelname)
                np.save(model_dir + 'Loss.npy', lossall)
            step += 1

            if step > iteration:
                break
        print("one epoch pass")
        epoch = epoch + 1
    np.save(model_dir + '/Loss.npy', lossall)
    
train()
