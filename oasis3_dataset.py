import pathlib
import torch.utils.data
import torchvision
import os
import numpy as np
import collections
import skimage.draw
import pandas as pd
import nibabel as nib
from skimage.transform import resize
import torch.utils.data as data
import matplotlib.pyplot as plt
import cv2
import json

IMG_PX_SIZE = 112
HM_SLICES = 32

ROTATING_PATIENTS = ['OAS30012_scan1', 'OAS30012_scan2', 'OAS30020_scan1', 'OAS30020_scan2',
            'OAS30022_scan1', 'OAS30032_scan1', 'OAS30038_scan1', 'OAS30040_scan1', 'OAS30041_scan1',
            'OAS30048_scan1', 'OAS30059_scan1', 'OAS30062_scan1', 'OAS30072_scan1', 'OAS30073_scan1',
            'OAS30094_scan1', 'OAS30094_scan2', 'OAS30103_scan1', 'OAS30106_scan1', 'OAS30118_scan1',
            'OAS30119_scan1', 'OAS30120_scan1', 'OAS30121_scan1', 'OAS30131_scan1', 'OAS30131_scan2',
            'OAS30141_scan1', 'OAS30153_scan1', 'OAS30156_scan1', 'OAS30157_scan1', 'OAS30157_scan2',
            'OAS30188_scan1', 'OAS30192_scan1', 'OAS30194_scan1', 'OAS30194_scan2', 'OAS30203_scan1',
            'OAS30203_scan2', 'OAS30216_scan1', 'OAS30234_scan1', 'OAS30237_scan1', 'OAS30240_scan1',
            'OAS30241_scan1', 'OAS30249_scan1', 'OAS30256_scan1', 'OAS30259_scan1', 'OAS30259_scan2', 
            'OAS30278_scan1', 'OAS30284_scan1', 'OAS30288_scan1', 'OAS30203_scan1', 'OAS30297_scan1',
            'OAS30303_scan1', 'OAS30314_scan1', 'OAS30314_scan2', 'OAS30322_scan1', 'OAS30322_scan2', 
            'OAS30331_scan1', 'OAS30343_scan1', 'OAS30343_scan2', 'OAS30347_scan1', 'OAS30347_scan2',
            'OAS30365_scan1', 'OAS30365_scan2', 'OAS30367_scan1', 'OAS30369_scan1', 'OAS30379_scan1',
            'OAS30381_scan1', 'OAS30393_scan1', 'OAS30393_scan2', 'OAS30408_scan1', 'OAS30411_scan1',
            'OAS30412_scan1', 'OAS30420_scan1', 'OAS30444_scan1', 'OAS30446_scan1', 'OAS30452_scan1',
            'OAS30455_scan1', 'OAS30459_scan1', 'OAS30459_scan2', 'OAS30463_scan1', 'OAS30463_scan2',
            'OAS30470_scan1', 'OAS30494_scan1', 'OAS30505_scan1', 'OAS30505_scan2', 'OAS30506_scan1',
            'OAS30516_scan1', 'OAS30530_scan1', 'OAS30530_scan2', 'OAS30539_scan1', 'OAS30539_scan2',
            'OAS30558_scan1', 'OAS30563_scan1', 'OAS30563_scan2', 'OAS30564_scan1', 'OAS30564_scan2',
            'OAS30565_scan1', 'OAS30581_scan1', 'OAS30588_scan1', 'OAS30614_scan1', 'OAS30627_scan1',
            'OAS30660_scan1', 'OAS30662_scan1', 'OAS30662_scan2', 'OAS30669_scan1', 'OAS30675_scan1',
            'OAS30676_scan1', 'OAS30689_scan1', 'OAS30698_scan1', 'OAS30698_scan2', 'OAS30699_scan1',
            'OAS30700_scan1', 'OAS30709_scan1', 'OAS30718_scan1', 'OAS30720_scan1', 'OAS30720_scan2',
            'OAS30724_scan1', 'OAS30736_scan1', 'OAS30700_scan1', 'OAS30742_scan1', 'OAS30745_scan1',
            'OAS30776_scan1', 'OAS30777_scan1', 'OAS30784_scan1', 'OAS30786_scan1', 'OAS30787_scan1',
            'OAS30787_scan2', 'OAS30791_scan1', 'OAS30794_scan1', 'OAS30803_scan1', 'OAS30803_scan2',
            'OAS30825_scan1', 'OAS30826_scan1', 'OAS30828_scan1', 'OAS30829_scan1', 'OAS30829_scan2',
            'OAS30837_scan1', 'OAS30841_scan1', 'OAS30852_scan1', 'OAS30852_scan2', 'OAS30855_scan1',
            'OAS30860_scan1', 'OAS30860_scan2', 'OAS30863_scan1', 'OAS30867_scan1', 'OAS30867_scan2',
            'OAS30870_scan1', 'OAS30872_scan1', 'OAS30876_scan1', 'OAS30878_scan1', 'OAS30919_scan1',
            'OAS30920_scan1', 'OAS30923_scan1', 'OAS30923_scan2', 'OAS30932_scan1', 'OAS30936_scan1',
            'OAS30936_scan2', 'OAS30941_scan1', 'OAS30946_scan1', 'OAS30947_scan1', 'OAS30960_scan1',
            'OAS30960_scan2', 'OAS30963_scan1', 'OAS30964_scan1', 'OAS30965_scan1', 'OAS30965_scan2',
            'OAS30969_scan1', 'OAS30972_scan1', 'OAS30984_scan1', 'OAS30984_scan1']


def need_to_rotate(p, scan_id):
    # print(p[scan_id])
    # p[scan_id + '_rotate'] = 0
    # return
    # json_path = str(p[scan_id][:-6]) + "json"

    patient = str(p['id']) + "_" + str(scan_id)
    if patient in ROTATING_PATIENTS:
        p[scan_id + '_rotate'] = 1
    return

    with open(json_path) as f:
        json_file = json.load(f)
        try:
            if json_file['ConversionSoftware'] == "dcm2nii":
                p[scan_id + '_rotate'] = 1
                print(p['id'], " ", scan_id)
        except:
            print("can't find orientation for: {}, {}".format(p['id'], scan_id))


class OASIS3Dataset(torch.utils.data.Dataset):
    def __init__(self, filename="data_test.csv",
                 split="train",
                 mean=0., std=1.,
                 max_length=9,
                 crops=1,
                 pad=None,
                 target_transform=None):

        self.split = split
        self.mean, self.std = mean, std
        self.max_length = max_length
        self.crops = crops
        self.pad = pad
        self.target_transform = target_transform


        # Read in the data
        data = pd.read_csv(filename, index_col=0)
        rm = False
        try:
            remove = pd.read_csv("remove.csv").id.values
            # print(remove)
            rm = True
        except:
            print("Not removing any patients")

        #select the split (train, val, test)
        data = data.loc[data['split'] == split]
        
        patients = []
        for i in data.index:
            if rm == True:
                if i in remove:
                    continue
            p = {}
            p['id'] = i
            # print(i)
            n_scans =         data.loc[i, 'num_scans']
            p['num_scans']  = n_scans
            p['label']      = data.loc[i, 'label']
            p['scan1']      = data.loc[i, 'scan1']
            need_to_rotate(p, 'scan1')
            p['scan2']      = data.loc[i, 'scan2'] if n_scans >=2 else 'None'
            if p['scan2'] != 'None':
                need_to_rotate(p, 'scan2')
            p['scan3']      = data.loc[i, 'scan3'] if n_scans >=3 else 'None'
            if p['scan3'] != 'None':
                need_to_rotate(p, 'scan3')
            p['scan4']      = data.loc[i, 'scan4'] if n_scans >=4 else 'None'
            if p['scan4'] != 'None':
                need_to_rotate(p, 'scan4')
            p['scan5']      = data.loc[i, 'scan5'] if n_scans >=5 else 'None'
            if p['scan5'] != 'None':
                need_to_rotate(p, 'scan5')
            p['scan6']      = data.loc[i, 'scan6'] if n_scans >=6 else 'None'
            if p['scan6'] != 'None':
                need_to_rotate(p, 'scan6')
            p['scan7']      = data.loc[i, 'scan7'] if n_scans >=7 else 'None'
            if p['scan7'] != 'None':
                need_to_rotate(p, 'scan7')
            p['scan8']      = data.loc[i, 'scan8'] if n_scans >=8 else 'None'
            if p['scan8'] != 'None':
                need_to_rotate(p, 'scan8')
            p['scan9']      = data.loc[i, 'scan9'] if n_scans ==9 else 'None'
            if p['scan9'] != 'None':
                need_to_rotate(p, 'scan9')


            patients.append(p)

        self.patients = patients#[0:100]
        # print(patients[0:5])




    def __getitem__(self, index):
        """Returns one data instance (scans, label)."""
        patient = self.patients[index]

        scans = []
        found_one = False

        for i in range(1):#patient['num_scans']):
            scan_id = 'scan' + str(i+1)
            # print(patient['id'], " : ", patient[scan_id])
            scan = nib.load(patient[scan_id]).get_data()
            try:
                r = patient[scan_id + "_rotate"]
                # if r == 1: 
                #     scan = np.transpose(scan, (2, 0, 1))
                    # print("rotating patient id " + patient['id'])
            except:
                r = 0
            # print(patient['id'], " ", scan.shape)
            if scan.shape[0] < 16:
                continue
            scan = np.transpose(scan, (2, 0, 1)) # slice # first
            
            scan = [cv2.resize(np.array(each_silce), (IMG_PX_SIZE, IMG_PX_SIZE)) for\
                                each_silce in scan]
            # print(len(scan))
            # print(scan[0].shape)
            scan = np.array(scan, dtype=np.float32)
            # print(scan.shape)
            scan = np.transpose(scan, (1, 2, 0)) # height first
            scan = [cv2.resize(np.array(each_silce), (IMG_PX_SIZE, HM_SLICES)) for\
                                each_silce in scan]
            scan = np.array(scan, dtype=np.float32)
            scan = np.transpose(scan, (2, 0, 1)) # slice # first

            # print(scan.shape)
            scan = (scan - scan.min()) / (scan.max() - scan.min())
            # scan = (scan - self.mean)/ self.std
            found_one = True

            # scan = x.expand(39, 3, 20, 256, 256)
           
            # print(type(scan), scan.shape)
            scans.append(scan)
        # if found_one == False:
        #     print()
        #     print(patient['id'])
        #     print()
        scans = torch.tensor(scans).float()
        scans = scans.unsqueeze(0)
        # print()
        # print("get item: ", scans.shape)
        # print()
        if scans.shape[1] != 0:
            scans = scans.expand(3, scans.shape[1], IMG_PX_SIZE, IMG_PX_SIZE, HM_SLICES)
        
        label = torch.tensor(patient['label'])
        patient_id = patient['id']
        
        # print(type(scans), scans.shape)
        
        return scans, label, patient_id




    def __len__(self):
        return len(self.patients)

    def get_number_of_samples(self):
        return self.__len__()


def _defaultdict_of_lists():
    return collections.defaultdict(list)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
        We should build custom collate_fn rather than using default collate_fn, 
        because merging caption (including padding) is not supported in default.
        Args:
            data: list of tuple (tensor of images, label). 
                - images: torch tensor of shape (256, 256, 36).
                - caption: torch tensor of shape (?); variable length.
        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[0].shape[1], reverse=True)

    images, labels, patient = zip(*data)
    

    # Merge labels (from tuple of 1 tensor to 2D tensor).
    labels = torch.stack(labels, 0)

    # Merge images (from tuple of 4D tensor to 6D tensor).
    lengths = [img.shape[1] for img in images]
    total_length = sum(lengths)
    batch_size = len(images)
    # print(lengths)
    # print(images[0].dtype)

    list_of_imgs = []
    for i, img in enumerate(images):
        end = lengths[i]
        # print(end)
        for j in range(end):
            list_of_imgs.append(img[:, j, :, :, :])
            # print(list_of_imgs[i].shape)
    targets = torch.stack(list_of_imgs)
    # print(targets.shape)
    targets = targets.permute(0, 1, 4, 2, 3)
    # targets = torch.autograd.Variable(targets.var)
    lengths = torch.tensor(lengths)

    return targets, labels, lengths#, patient

def get_mean_and_std(dataset, split, samples=100):
    if len(dataset) > samples:
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), samples, replace=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    n = 0
    mean = 0.
    std = 0.

    split = 'Find mean and std of ' + split + 'set'
    for(i, (x, *_)) in enumerate(dataloader):
        if len(x.shape) != 6:
            continue
        x = x[:, :, 0, :, :, :]
        # print(x.shape)
        x = x.transpose(0, 1).contiguous().view(1, -1)
        n += 1
        mean += x.mean(dim=1).numpy()
        std += x.std(dim=1).numpy()
        
    # print(mean, std)
    mean /= n
    std /= n

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std

def get_oasis3_datasets():

    mean, std = get_mean_and_std(OASIS3Dataset(split="train"), 'train')
    print(mean, " ",std)

    kwargs = {
              "mean": mean,
              "std": std
              }

    dataset_train = OASIS3Dataset(split="train", **kwargs)

    # mean, std = get_mean_and_std(OASIS3Dataset(split="test"), 'test')
    # print(mean, " ",std)

    # kwargs = {
    #           "mean": mean,
    #           "std": std,
    #           }

    dataset_test = OASIS3Dataset(split="val", **kwargs)

    return dataset_train, dataset_test

def main():
    dataset, test_data = get_oasis3_datasets()
    # very first line missing!!
    # x, y = dataset[2]
    # print(type(x), " ", x.shape)
    trainloader = data.DataLoader(dataset, batch_size=1, \
                                    shuffle = False, num_workers=1,\
                                    drop_last=True,  collate_fn=collate_fn)

    for (i, (inputs, labels, lengths, patient)) in enumerate(trainloader):
        print(patient)
        print(lengths)
        print(type(inputs), " ", inputs.size())
        print(type(labels), " ", labels)
        for j in range(inputs.shape[0]):
            imgs = inputs[j, :, :, :, :].permute(1, 0, 2, 3)
            imgs = torchvision.utils.make_grid(imgs, nrow=8).permute(1, 2, 0)
            plt.imshow(imgs)
            plt.savefig("imgs_prep_train/patient{}_visit{}_({})".format(patient, j+1, labels[0].item()))
        if i == 50:
            break





if __name__ == "__main__":
    main()
