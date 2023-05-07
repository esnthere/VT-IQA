# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy import io as sio
import torch.utils.data
import torchvision.models as models
import pandas as pd
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
import os
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torchvision.models as models
from functools import partial

from my_vision_transformer2 import VisionTransformer, _cfg
from my_safusion import VisionTransformer as VisionTransformer2
from timm.models.vision_transformer import VisionTransformer as VisionTransformer3
from timm.models.registry import register_model
import matplotlib.pyplot as plt

class Mydataset(Dataset):
    def __init__(self, imgs,imgs2, labels):
        self.imgs = imgs
        self.imgs2 = imgs2
        self.labels = torch.FloatTensor(labels)

    def __getitem__(self, index):
        if self.imgs[0].shape[2]==244:
            img=self.imgs[index, :, 10:10 + 224, 10:10 + 224]
        else:
            img = self.imgs[index]

        img2= self.imgs2[index]
        return torch.from_numpy(img), torch.from_numpy(img2), self.labels[index]

    def __len__(self):
        return (self.imgs).shape[0]

def test(model,model2, model3,test_loader, epoch, device, all_test_loss):
    model.eval()
    model2.eval()
    model3.eval()

    op = []
    tg = []

    for batch_idx, (data, data2, target) in enumerate(test_loader):
        data, data2, target = data.to(device), data2.to(device), target.to(device)

        data_pt = data.clone()
        data = data.float()
        data /= 255
        data[:, 0] -= 0.485
        data[:, 1] -= 0.456
        data[:, 2] -= 0.406
        data[:, 0] /= 0.229
        data[:, 1] /= 0.224
        data[:, 2] /= 0.225


        data.requires_grad = True
        output = model3(data)
        output.sum().backward()
        g1 = data.grad
        # g1 = torch.abs(g1)
        g1 = torch.sum(g1, 1).detach()
        cam = F.avg_pool2d(g1, (9, 9), stride=9).cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        data.requires_grad = False

        nm = 7
        patch_sz = int(224 / nm)

        for n in range(cam.shape[0]):
            for i in range(nm):
                for j in range(nm):
                    maxn = -10
                    for x in range(3):
                        for y in range(3):
                            if cam[n, i * 3 + x, j * 3 + y] > maxn:
                                maxn=cam[n, i * 3 + x, j * 3 + y]
                                i2 = int((1.5 + i * 3 + x) / 24 * data2.shape[2])
                                j2 = int((1.5 + j * 3 + y) / 24 * data2.shape[2])
                    data_pt[n, :, i * patch_sz:i * patch_sz + patch_sz, j * patch_sz:j * patch_sz + patch_sz] = data2[n, :, i2:i2 + patch_sz, j2:j2 + patch_sz]

        data_pt = data_pt.float()
        data_pt /= 255
        data_pt[:, 0] -= 0.485
        data_pt[:, 1] -= 0.456
        data_pt[:, 2] -= 0.406
        data_pt[:, 0] /= 0.229
        data_pt[:, 1] /= 0.224
        data_pt[:, 2] /= 0.225


        with torch.no_grad():
            ftsall = model(torch.cat((data, data_pt), 0))
            x2 = torch.zeros((data.shape[0], 2, ftsall.shape[1]))
            x2[:, 0] = ftsall[:data.shape[0]]
            x2[:, 1] = ftsall[data.shape[0]:]
            output = model2(x2)

        op = np.concatenate((op, output[:, 0].cpu().numpy()))
        tg = np.concatenate((tg, target[:, 0].cpu().numpy()))



    print( 'ALL Pearson:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"))
    print( 'ALL Spearman:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman"))
    return all_test_loss, pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson")



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda")

    model = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    model = nn.DataParallel(model.to(device))
    model.load_state_dict(torch.load("VTIQA_org_backbone.pt"))


    model2 = VisionTransformer2(num_patches=2, embed_dim=384, depth=2, num_heads=1, mlp_ratio=1, qkv_bias=True,drop_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model2.default_cfg = _cfg()
    model2.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(384 * 3, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.Linear(8, 1))

    model2 = nn.DataParallel(model2.to(device))
    model2.load_state_dict(torch.load("VTIQA_org_safusion.pt"))

    model3 = VisionTransformer3(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model3.default_cfg = _cfg()
    model3.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(384, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
        nn.ReLU(),
        nn.Linear(8, 1))
    model3 = nn.DataParallel(model3.to(device))
    model3.load_state_dict(torch.load("VTIQA_org_slection.pt"))


    batch_size = 64
    num_workers_test = 0

#########################################################


    all_data = sio.loadmat('E:\Database\IQA Database\## KonIQ-10k Image Database\Koniq_244.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)

    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    del all_data

    all_data = np.load('E:\Database\IQA Database\## KonIQ-10k Image Database\Koniq_768.npz')
    X2 = all_data['X']
    Xtest2 = all_data['Xtest']
    del all_data


    test_dataset = Mydataset(Xtest,Xtest2, Ytest)

    test_loader = DataLoader(test_dataset, batch_size=batch_size*1, shuffle=False, num_workers=num_workers_test, pin_memory=True)

    print("koniq Test Results:")
    all_test_loss = []
    test(model, model2,model3, test_loader, -1, device, all_test_loss)

    # #
# # #####################################
    all_data = sio.loadmat('E:\Database\LIVEW\livew_244.mat')

    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Y = Y / 25 + 1

    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    Ytest = Ytest / 25 + 1
    del all_data

    all_data = sio.loadmat('E:\Database\LIVEW\livew_768.mat')
    X2 = all_data['X']
    Xtest2 = all_data['Xtest']
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), 0), np.concatenate((X2, Xtest2), 0),np.concatenate((Y, Ytest), 0))


    test_loader = DataLoader(test_dataset, batch_size=batch_size*1, shuffle=False, num_workers=num_workers_test, pin_memory=True)
    print("Livew Test Results:")

    all_test_loss = []
    test(model, model2,model3, test_loader, -1, device, all_test_loss)
    ######################################################

    all_data = sio.loadmat('E:\Database\CID2013（database）\cid_244.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = (Y + 10) / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = (Ytest + 10) / 25 + 1
    del all_data

    all_data = sio.loadmat('E:\Database\CID2013（database）\cid_768.mat')
    X2 = all_data['X']
    Xtest2 = all_data['Xtest']
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), 0), np.concatenate((X2, Xtest2), 0), np.concatenate((Y, Ytest), 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 1, shuffle=False, num_workers=num_workers_test, pin_memory=True)

    all_test_loss = []
    print("CID Test Results:")
    test(model, model2,model3, test_loader, -1, device, all_test_loss)

    #######################################################

    all_data = sio.loadmat('E:\Database\RBID\\rbid_244.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y * 0.8 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest * 0.8 + 1
    del all_data

    all_data = sio.loadmat('E:\Database\RBID\\rbid_768.mat')
    X2 = all_data['X']
    Xtest2 = all_data['Xtest']
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), 0), np.concatenate((X2, Xtest2), 0),
                             np.concatenate((Y, Ytest), 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                              pin_memory=True)

    all_test_loss = []
    print("RBID Test Results:")
    test(model, model2, model3, test_loader, -1, device, all_test_loss)
    #######################################################

    all_data = np.load('E:\Database\SPAQ\spaq_244.npz')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y.reshape(Y.shape[0], 1)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest.reshape(Ytest.shape[0], 1)
    Ytest = Ytest / 25 + 1
    del all_data

    all_data = np.load('E:\Database\SPAQ\spaq_768.npz')
    X2 = all_data['X']
    Xtest2 = all_data['Xtest']
    del all_data

    test_dataset = Mydataset(np.concatenate((X, Xtest), 0), np.concatenate((X2, Xtest2), 0),np.concatenate((Y, Ytest), 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 1, shuffle=False, num_workers=num_workers_test,pin_memory=True)

    all_test_loss = []
    print("SPAQ Test Results:")
    test(model, model2,model3, test_loader, -1, device, all_test_loss)



if __name__ == '__main__':
    main()

