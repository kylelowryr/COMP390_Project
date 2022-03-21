import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random

class MultiviewImgDataset2(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20, shuffle=True, cover =0,start=0, disturb=0):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.cover = cover
        self.start = start
        self.disturb = disturb
        set_ = root_dir.split('/')[-1]


        parent_dir = root_dir.rsplit('/',2)[0]

        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            ## Select subset for different number of views
            stride = int(20/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new

        if start != 0:
            filepaths_new = []
            for i in range(int(len(self.filepaths)/num_views)):
                filepaths_new.extend(self.filepaths[i*num_views+start:(i+1)*num_views])
                filepaths_new.extend(self.filepaths[i*num_views:i*num_views+start])
            #print(filepaths_new)
            self.filepaths = filepaths_new
        else:
            #print(self.filepaths)
            pass

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-2]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            if self.cover == -1:
                im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
                if self.transform:
                    im = self.transform(im)
                if self.disturb > 0:
                    randindex1 = random.sample(range(56, 178), self.disturb)
                    #print('random1 ',randindex1)
                    randindex2 = random.sample(range(56, 178), self.disturb)
                    #print('random2 ', randindex2)
                    for i in range(self.disturb):
                        im[0][randindex1[i]][randindex2[i]] = 2.2489
                        im[1][randindex1[i]][randindex2[i]] = 2.4286
                        im[2][randindex1[i]][randindex2[i]] = 2.6400
                imgs.append(im)
            else:
                if i == self.cover:
                    im = Image.open('all_white.png').convert('RGB')
                    #self.filepaths[idx * self.num_views + i] = 'all_black.png'
                else:
                    im = Image.open(self.filepaths[idx * self.num_views + i]).convert('RGB')
                if self.transform:
                    im = self.transform(im)
                imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])



class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        set_ = root_dir.split('/')[-1]
        print(set_)
        parent_dir = root_dir.rsplit('/',2)[0]
        print(parent_dir)
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-2]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')


        # if self.transform:
            # im = self.transform(im)
        if self.transform:
            im = self.transform(im)


        return (class_id, im, path)



def generate_dataset(num_views,args):
    val_list = []
    val_dataset = MultiviewImgDataset2(args.val_path, scale_aug=False, rot_aug=False, shuffle=False,
                                      num_views=args.num_views, cover=-1, start=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    val_list.append(val_loader)
    for i in range(num_views):
        num_str = str(i)
        name_dataset = 'val_dataset_cover'+num_str
        name_loader = 'val_loader_cover'+num_str
        locals()[name_dataset] = MultiviewImgDataset2(args.val_path, scale_aug=False, rot_aug=False, shuffle=False,
                                                 num_views=args.num_views, cover=i, start=0)
        #print(locals()[name_dataset][0])
        locals()[name_loader]= torch.utils.data.DataLoader(locals()[name_dataset], batch_size=1, shuffle=False, num_workers=0)
        val_list.append(locals()[name_loader])
    return val_list