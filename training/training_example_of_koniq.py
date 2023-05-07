import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import math
from scipy import io as sio
import torch.utils.data
import torchvision.models as models
from imgaug import augmenters as iaa
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import time
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from skimage import io
import os
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torchvision.models as models
from functools import partial

from my_vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from prefetch_generator import BackgroundGenerator
import matplotlib.pyplot as plt
import lmdb


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Mydataset(Dataset):
    def __init__(self, imgs,imgs2, labels):
        self.imgs = imgs
        self.imgs2 = imgs2
        self.labels = torch.FloatTensor(labels)
    def __getitem__(self, index):
        return torch.from_numpy(self.imgs[index]), torch.from_numpy(self.imgs2[index]),self.labels[index]
    def __len__(self):
        return (self.imgs).shape[0]


def detect_loss(pred, label):
    # loss = torch.mean(torch.pow(pred - label, 2)) + torch.mean(torch.abs(pred - label))
    # loss =  torch.mean(torch.abs(pred - label))
    loss = torch.mean(torch.pow(pred - label, 2)) + 0.2*torch.mean(torch.abs(pred - label))
    return loss


def train(model, train_loader, optimizer, epoch, device, all_train_loss):
    model.train()
    # model.apply(fix_bn)
    st = time.time()
    op=[]
    tg=[]
    for batch_idx, (data,data2, target) in enumerate(train_loader):
        data, data2, target = data.to(device), data2.to(device), target.to(device)
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd_ps = torch.randint(20, (3,))
        data = data[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
        if rd_ps[1] < 10:
            data = torch.flip(data, dims=[3])

        data_pt=data.clone()
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd = torch.randint(5,(1,))[0]-2
        nm = 7
        # nm=nm+rd
        patch_sz = int(224 / nm)

        scale1=int(768/nm)
        scale2 = int(768 / nm)
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd_ps1 = torch.randint(scale1-patch_sz, (nm,))
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd_ps2 = torch.randint(scale2-patch_sz, (nm,))
        
        for i in range(nm):
            for j in range(nm):
                data_pt[:,:,i*patch_sz:i*patch_sz+patch_sz,j*patch_sz:j*patch_sz+patch_sz]= data2[:,:,i*scale2+rd_ps2[i]:i*scale2+rd_ps2[i]+patch_sz,j*scale1+rd_ps1[j]:j*scale1+rd_ps1[j]+patch_sz]

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
        optimizer.zero_grad()
        # fts1 = model.module.forward_features(data)
        # fts2 = model.module.forward_features(data_pt)
        # fts=torch.cat((fts1,fts2),1)
        # output=model.module.head(fts)
        output = model(data,data_pt)

        loss = F.l1_loss(output, target)
        all_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        op = np.concatenate((op, output[:, 0].detach().cpu().numpy()))
        tg = np.concatenate((tg, target[:, 0].cpu().numpy()))
        p1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson")
        s1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman")

        if batch_idx % 100 == 0:
            print('Train Epoch:{} [({:.0f}%)]\t Loss: {:.4f}  Pearson:{:.4f} Spearman:{:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item(), p1, s1))

    print( 'Train ALL Pearson:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"))
    print( 'Train  ALL Spearman:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman"))
    return all_train_loss


def test(model, test_loader, epoch, device, all_test_loss):
    model.eval()
    test_loss = 0
    pearson = 0
    spearman = 0
    op = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, data2,target) in enumerate(test_loader):
            data, data2, target = data.to(device), data2.to(device), target.to(device)

            data = data[:, :,10:10+ 224, 10:10+ 224]

            data_pt = data.clone()
            nm = 7
            patch_sz = int(224 / nm)
            scale1 = int(768 / nm)
            scale2 = int(768 / nm)
            rd_ps1 =int(scale1/2-patch_sz/2)
            rd_ps2  =int(scale2/2-patch_sz/2)

            for i in range(nm):
                for j in range(nm):
                    data_pt[:, :, i * patch_sz:i * patch_sz + patch_sz, j * patch_sz:j * patch_sz + patch_sz] = data2[:,:,i * scale2 +rd_ps2:i * scale2 +rd_ps2 + patch_sz, j * scale1 +rd_ps1:j * scale1 +rd_ps1 + patch_sz]


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

            # fts1 = model.module.forward_features(data)
            # fts2 = model.module.forward_features(data_pt)
            # fts = torch.cat((fts1, fts2), 1)
            # output = model.module.head(fts)
            output = model(data, data_pt)

            loss = F.l1_loss(output, target)
            all_test_loss.append(loss)
            test_loss += loss
            op = np.concatenate((op, output[:, 0].cpu().numpy()))
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))
            p1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson")
            s1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman")
            pearson += p1
            spearman += s1
            if batch_idx % 100 == 0:
                print('Test Epoch:{} [({:.0f}%)]\t Loss: {:.4f}  Pearson:{:.4f} Spearman:{:.4f}'.format(
                    epoch, 100. * batch_idx / len(test_loader), loss.item(), p1, s1))

    test_loss /= (batch_idx + 1)
    pearson /= (batch_idx + 1)
    spearman /= (batch_idx + 1)
    print('Test : Loss:{:.4f} '.format(test_loss))
    print("pearson：", pearson, 'ALL Pearson:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"))
    print("spearman：", spearman, 'ALL Spearman:',pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman"))
    return all_test_loss, pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"), pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    device = torch.device("cuda")


    all_data = sio.loadmat('E:\Database\IQA Database\## KonIQ-10k Image Database\Koniq_244.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1,0)

    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1,0)
    del all_data

    all_data = np.load('E:\Database\IQA Database\## KonIQ-10k Image Database\Koniq_768.npz')
    X2 = all_data['X']
    Xtest2 = all_data['Xtest']
    del all_data



    best_pl=[]
    best_sp=[]
    for i in range(10):

        model = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model.default_cfg = _cfg()
        model.load_state_dict(torch.load("deit_small_patch16_224.pth")['model'])
        model.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(384 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1))

        for param in model.parameters():
            param.requires_grad = False

        for param in model.head.parameters():
            param.requires_grad = True

        model = nn.DataParallel(model.to(device))

        # model.load_state_dict(torch.load("Koniq_efctb0_080_aug20_1.pt"))
        X = np.concatenate((X, Xtest), axis=0)
        Y = np.concatenate((Y, Ytest), axis=0)
        X2 = np.concatenate((X2, Xtest2), axis=0)
        ind=np.arange(0,X.shape[0])
        np.random.seed(i)
        np.random.shuffle(ind)

        Xtest = X[ind[int(len(ind) * 0.8):]]
        Ytest = Y[ind[int(len(ind) * 0.8):]]
        Xtest2=X2[ind[int(len(ind) * 0.8):]]
        X = X[ind[:int(len(ind) * 0.8)]]
        Y = Y[ind[:int(len(ind) * 0.8)]]
        X2 = X2[ind[:int(len(ind) * 0.8)]]

        train_dataset = Mydataset(X,X2, Y)
        test_dataset = Mydataset(Xtest, Xtest2,Ytest)

        max_plsp=-1
        min_loss = 1e8
        lr = 0.01
        weight_decay = 1e-3
        batch_size = 32*6
        epochs = 2000
        num_workers_train = 0
        num_workers_test = 0
        ct=0


        train_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_train,pin_memory=True)
        test_loader = DataLoaderX(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_test,pin_memory=True)

        all_train_loss = []
        all_test_loss = []
        all_test_loss, _,_ = test(model, test_loader, -1, device, all_test_loss)
        ct = 0
        lr = 0.01
        max_plsp = -2

        for epoch in range(epochs):
            print(lr)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                  weight_decay=weight_decay)
            ct += 1
            start = time.time()
            all_train_loss = train(model, train_loader, optimizer, epoch, device, all_train_loss)
            print(time.time() - start)
            all_test_loss, plsp,_ = test(model, test_loader, epoch, device, all_test_loss)
            print("time:", time.time() - start)
            if epoch == 10:
                for param in model.parameters():
                    param.requires_grad = True
                lr = 0.001

            if max_plsp < plsp:
                save_nm = 'Koniq244_deit3fc_with_768patch2_10times_1_'+str(i)+'.pt'
                max_plsp = plsp
                torch.save(model.state_dict(), save_nm)
                ct = 0

            if epoch  ==20:
                lr= 0.005
            if epoch == 30:
                lr = 0.03
                ct = 1

            if ct > 20 and epoch > 20:
                model.load_state_dict(torch.load(save_nm))
                lr *= 0.3
                ct = 0
                if lr<5e-5:
                    all_test_loss, pl,sp = test(model, test_loader, -1, device, all_test_loss)
                    best_pl .append(pl)
                    best_sp .append(sp)
                    break
        print('PLCC:' ,best_pl,'SRCC:',best_sp)
        print('Split:',i,'Median PLCC:', np.median(np.array(best_pl)), 'SRCC:',np.median(np.array(best_sp)))

if __name__ == '__main__':
    main()

