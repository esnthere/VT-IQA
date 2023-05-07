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
import matplotlib.pyplot as plt
import cv2
import torchvision.models as models
from functools import partial

from my_vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model


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









def test(model, test_loader, epoch, device, all_test_loss):
    model.eval()
    op = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, data2, target) in enumerate(test_loader):
            data, data2, target = data.to(device), data2.to(device), target.to(device)

            data_pt = data.clone()
            nm = 7
            # sz=max(data2.shape[2],data2.shape[3])
            # nm=round(nm*1024/sz)
            patch_sz = int(224 / nm)
            scale1 = int(data2.shape[3] / nm)
            scale2 = int(data2.shape[2] / nm)

            rd_ps1 = int(scale1 / 2 - patch_sz / 2)
            rd_ps2 = int(scale2 / 2 - patch_sz / 2)

            for i in range(nm):
                for j in range(nm):
                    data_pt[:, :, i * patch_sz:i * patch_sz + patch_sz, j * patch_sz:j * patch_sz + patch_sz] = data2[:, :,i * scale2 + rd_ps2:i * scale2 + rd_ps2 + patch_sz, j * scale1 + rd_ps1: j * scale1 + rd_ps1 + patch_sz]
            # plt.imshow(data_pt[0].cpu().numpy().transpose(1, 2, 0))
            data_pt = data_pt.float()
            data_pt /= 255
            data_pt[:, 0] -= 0.485
            data_pt[:, 1] -= 0.456
            data_pt[:, 2] -= 0.406
            data_pt[:, 0] /= 0.229
            data_pt[:, 1] /= 0.224
            data_pt[:, 2] /= 0.225

            data = data.float()
            data /= 255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225


            output = model(data, data_pt)
            op = np.concatenate((op, output[:, 0].cpu().numpy()))
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))

    print( 'ALL Pearson:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"))
    print('ALL Spearman:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman"))
    return 0




def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda")

    model = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    model.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(384*2, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
        nn.ReLU(),
        nn.Linear(8, 1))

    model = nn.DataParallel(model.to(device), device_ids=[0])

    model.load_state_dict(torch.load('VTIQA_spaq.pt'))

    batch_size = 64
    num_workers_test = 0


# # #####################################
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

    X = np.concatenate((X, Xtest), axis=0)
    Y = np.concatenate((Y, Ytest), axis=0)
    X2 = np.concatenate((X2, Xtest2), axis=0)
    ind = np.arange(0, X.shape[0])
    np.random.seed(0)
    np.random.shuffle(ind)

    Xtest = X[ind[int(len(ind) * 0.8):]]
    Ytest = Y[ind[int(len(ind) * 0.8):]]
    Xtest2 = X2[ind[int(len(ind) * 0.8):]]


    test_dataset = Mydataset(Xtest, Xtest2, Ytest)

    test_loader = DataLoader(test_dataset, batch_size=batch_size*1, shuffle=False, num_workers=num_workers_test,pin_memory=True)
    print("SPAQ Test Results:")

    all_test_loss = []
    test(model,test_loader, -1, device, all_test_loss)

    return

if __name__ == '__main__':
    main()

