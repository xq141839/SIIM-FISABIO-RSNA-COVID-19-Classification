import os
import torch
import numpy as np
from skimage import io, transform, color
from torch.utils.data import Dataset
import albumentations as A
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
import time
import torch.nn.functional as F
from torch.utils.data import random_split
import random
from efficientnet_pytorch import EfficientNet as efnet
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

class ds_train(Dataset):
    def __init__(self,img_path,mask_path, df):
        self.img_path = list(img_path)
        self.match_id = df['id']
        self.df = df
        self.filename = []
        self.filename_mask = []
        self.labels = []
        for i in self.img_path:
            study_id = i.split("/")[2].split('.')[0]
            sty_class = self.df[self.match_id==study_id]
            if list(sty_class['Negative for Pneumonia'])[0] == 1:
                 self.labels.append(2)
            if list(sty_class['Typical Appearance'])[0] == 1:
                 self.labels.append(3)
            if list(sty_class['Indeterminate Appearance'])[0] == 1:
                 self.labels.append(1)
            if list(sty_class['Atypical Appearance'])[0] == 1:
                 self.labels.append(0)
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self,idx):
        image_path = self.img_path[idx]
        label = self.labels[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        my_transform1 = A.Compose([A.Resize(512,512),
                                   A.HorizontalFlip(p=0.5),
                                   A.ShiftScaleRotate(shift_limit=0,p=0.5),
                                   A.RandomBrightnessContrast(contrast_limit=0, p=0.5),
                                   A.CoarseDropout(p=0.5),
                                   A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229,0.224,0.225))
                                    ])
        aug = my_transform1(image=img)
        img = aug['image']
        my_tensor = transforms.ToTensor()
        img = my_tensor(img)
        return (img,label)

class ds_valid(Dataset):
    def __init__(self,img_path,mask_path, df):
        self.img_path = list(img_path)
        self.match_id = df['id']
        self.df = df
        self.filename = []
        self.filename_mask = []
        self.labels = []
        for i in self.img_path:
            study_id = i.split("/")[2].split('.')[0]
            sty_class = self.df[self.match_id==study_id]
            if list(sty_class['Negative for Pneumonia'])[0] == 1:
                 self.labels.append(2)
            if list(sty_class['Typical Appearance'])[0] == 1:
                 self.labels.append(3)
            if list(sty_class['Indeterminate Appearance'])[0] == 1:
                 self.labels.append(1)
            if list(sty_class['Atypical Appearance'])[0] == 1:
                 self.labels.append(0)
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self,idx):
        image_path = self.img_path[idx]
        label = self.labels[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        my_transform1 = A.Compose([A.Resize(512,512),
                                   A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229,0.224,0.225))
                                    ])
        aug = my_transform1(image=img)
        img = aug['image']
        my_tensor = transforms.ToTensor()
        img = my_tensor(img)
        return (img,label)

df = pd.read_csv('src/train_study_level.csv')
gkf  = GroupKFold(n_splits = 5)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups = df.id.tolist())):
    df.loc[val_idx, 'fold'] = fold

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    Auc_list = {'train': [], 'valid': []}
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_auc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            final_pro = []
            final_label = []
            final_pred = []
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                #print(inputs.shape)
                # forward
                outputs = model(inputs)
                #print(outputs)
                _, preds = torch.max(outputs.data, 1)
               
                loss1 = F.cross_entropy(outputs, labels)
                pro = F.softmax(outputs,dim=1).cpu().detach().numpy()
                label_np = labels.cpu().detach().numpy()
                preds_np = preds.cpu().detach().numpy()
                loss = loss1
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                loss_value = loss.item()
                running_loss += loss_value
                if len(final_pro) == 0:
                    final_pro = pro
                else:
                    final_pro = np.concatenate((final_pro, pro),0)
                if len(final_label) == 0:
                    final_label = label_np
                else:
                    final_label = np.concatenate((final_label, label_np),0)
                if len(final_pred) == 0:
                    final_pred = preds_np
                else:
                    final_pred = np.concatenate((final_pred, preds_np),0)
                running_corrects += torch.sum(preds == labels.data).float()


            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_auc = round(roc_auc_score(final_label, final_pro, average='weighted', multi_class='ovr'), 6) 
            

            print('{} Loss: {:.4f} AUC: {:.4f} ACC: {:.4f}'.format(
                phase, epoch_loss, epoch_auc, epoch_acc))
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc*100)
            Auc_list[phase].append(epoch_auc*100)
            print(confusion_matrix(final_label, final_pred))
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            if phase == 'valid':
                scheduler.step(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,Loss_list,Accuracy_list,Auc_list

    for i in range(1):
    fold_num = i
    valid_paths = 'src/study/' + df[df['fold'] == fold_num]['id'] + '.png' 
    train_paths = 'src/study/' + df[df['fold'] != fold_num]['id'] + '.png' 

    df2 = pd.read_csv('src/train_study_level.csv')
    train_data = ds_train(train_paths,None,df2)
    valid_data = ds_valid(valid_paths,None,df2)

    train_set = torch.utils.data.DataLoader(train_data,shuffle=True,batch_size=32, drop_last=True)
    valid_set = torch.utils.data.DataLoader(valid_data,shuffle=False,batch_size=32, drop_last=True)

    dataset_sizes = {'train':len(train_set.dataset),'valid':len(valid_set.dataset)}
    dataloaders = {'train':train_set,'valid':valid_set}
    
    model_ft = efnet.from_name('efficientnet-b7')

    state_dict = torch.load('src/pretrained_models/efficientnet-b7-dcc49843.pth')
    model_dict = model_ft.state_dict()
    pre_dict = {k: v for k, v in state_dict.items() if k in model_dict and '_fc' not in k}
    model_dict.update(pre_dict)
    model_ft.load_state_dict(model_dict)

    num_features = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_features,4)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
       #将batchsize 30 分配到N个GPU上运行
        model_ft = nn.DataParallel(model_ft)
    model_ft.to(device)
    optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.001)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,patience=3, min_lr=1e-6)

    model_ft,Loss_list,Accuracy_list,Auc_list = train_model(model_ft, None, optimizer_ft, exp_lr_scheduler,
                           num_epochs=1)


    #save the model
    name = 'model_b7_ce_' + str(i) +'.pth'
    torch.save(model_ft, name)

    #draw the plot
    x = range(0, len(Loss_list["valid"]))
    y1 = np.array(Loss_list["valid"])
    y2 = np.array(Loss_list["train"])
    plt.figure(figsize=(18,14))
    plt.subplot(211)
    plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="valid")
    plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
    plt.legend()
    plt.title('train and val loss vs. epoches')
    plt.ylabel('loss')

    plt.subplot(212)
    y3 = Accuracy_list["train"]
    y4 = Accuracy_list["valid"]
    plt.plot(x, y3, color="y", linestyle="-", marker=".", linewidth=1, label="train_acc")
    plt.plot(x, y4, color="g", linestyle="-", marker=".", linewidth=1, label="valid_acc")
    plt.legend()
    plt.title('train and valid accuracy')
    plt.ylabel('accuracy')
