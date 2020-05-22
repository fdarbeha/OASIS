import os
import argparse
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data

from pytorch_metric_learning import losses
from torchvideotransforms import video_transforms, volume_transforms

import sklearn.metrics
import numpy as np


from helper_functions import get_next_model_folder, inspect_model, reshape_images_cnn_input, reshape_rnn_input
from helper_functions import get_image_patch_tensor_from_volume_batch, write_csv_stats
from helper_classes import AverageMeter, GaussianBlur
from oasis3_dataset import get_oasis3_datasets, collate_fn
from model import construct_3d_enc, construct_rnn 

from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s=1
video_transform_list = [video_transforms.RandomRotation(5),
                        # video_transforms.RandomCrop((80, 80)),
                        # video_transforms.Resize(112, 112),
                        # video_transforms.RandomResize((112, 112)),
                        # video_transforms.RandomHorizontalFlip(),
                        # video_transforms.ColorJitter(0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s),
                        volume_transforms.ClipToTensor(3, 3)]

data_augment = video_transforms.Compose(video_transform_list)

# color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
# data_augment = transforms.Compose([transforms.ToPILImage(),
#                                    transforms.RandomResizedCrop(32),
#                                    transforms.RandomHorizontalFlip(),
#                                    transforms.RandomApply([color_jitter], p=0.8),
#                                    transforms.RandomGrayscale(p=0.2),
#                                    GaussianBlur(),
#                                    transforms.ToTensor()])

def dataloader(batch_size):
    trainloader, testloader = None, None
    if args.dataset.lower() == 'oasis3':
        
        trainset, testset = get_oasis3_datasets()

        print('TRAIN DATASET: ', len(trainset), trainset[0][0].shape)
        print('TEST DATASET: ', len(testset), testset[0][0].shape)
        
        trainloader = data.DataLoader(trainset, batch_size=batch_size, \
                                    shuffle = True, num_workers=5, drop_last=True,\
                                    collate_fn=collate_fn)
        testloader  = data.DataLoader(testset, batch_size=batch_size, \
                                    shuffle = False, num_workers=5, drop_last=True,\
                                    collate_fn=collate_fn)

    return trainloader, testloader

def optimizer(net, args):
    assert args.optimizer.lower() in ["sgd", "adam", "radam"], "Invalid Optimizer"

    if args.optimizer.lower() == "sgd":
           return optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta1, weight_decay=1e-4)
    elif args.optimizer.lower() == "adam":
           return optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer.lower() == "radam":
            return radam.RAdam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))


def load_model(output, epoch, model):
    epoch = "simclr_epoch_{}.pth".format(epoch)
    try:
        print("checkpoint: ", os.path.join(output, epoch))
        checkpoint = torch.load(os.path.join(output, epoch))
        model.load_state_dict(checkpoint)
        # optim.load_state_dict(checkpoint['opt_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_dict'])
        # epoch_resume = checkpoint["epoch"] + 1
        # bestLoss = checkpoint["best_loss"]
        # f.write("Resuming from epoch {}\n".format(epoch_resume))
    except FileNotFoundError:
        print("No checkpoint found\n")
    except: # saved model in nn.DataParallel
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def train_supervise(cnn, rnn, epoch, criterion, optimizer, trainloader, scaler):
    loss_meter = AverageMeter()
    running_loss = 0
    cnn.train()
    rnn.train()

    yhat = []
    y = []

    # for (i, (inputs, labels, lengths)) in enumerate(trainloader):
    for (i, (inputs, labels, lengths)) in enumerate(tqdm.tqdm(trainloader)):
        if inputs == None:
            continue
        # inputs = inputs.to(device)        
        labels = labels.to(device)
        lengths = lengths.to(device)

        # print("input shape: ", inputs.shape)
        # print("lengths: ", lengths)
        _, C, D, H, W = inputs.shape
        batch_size = lengths.shape[0]

        optimizer.zero_grad()

        x_1 = torch.zeros_like(inputs)
            
        for idx, x1 in enumerate(inputs):
                x1 = get_image_patch_tensor_from_volume_batch(x1)
                x_1[idx] = data_augment(x1)

        inputs = x_1.to(device)
                 
        with torch.set_grad_enabled(True):
            # with autocast():

            
            # inputs = reshape_images_cnn_input(inputs, lengths)
            # print("input of cnn shape ", inputs.shape)
            outputs = cnn(inputs)
            outputs = reshape_rnn_input(outputs, lengths, batch_size)
            # # outputs.to(device)
            # # print("input of rnn shape ",outputs.shape)
            outputs = rnn(outputs, lengths)
            # print(labels)
            # print()
            # print(outputs)
            loss = criterion(outputs.view(-1).float(), labels.float())

            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        outputs = outputs.to("cpu")

        loss_meter.update(loss.item())
        
        yhat.append(outputs.view(-1).detach().numpy())
        y.append(labels.to("cpu").numpy())

        # print("train loss: ", loss.item())
        running_loss += loss.item()

    y = np.reshape(y, len(y) * batch_size)
    yhat = np.reshape(yhat, len(yhat) * batch_size)
    # print(y)
    # print(yhat)

    auc = sklearn.metrics.roc_auc_score(y, yhat)
    return loss_meter.average(), auc

def eval_supervise(cnn, rnn, epoch, criterion, testloader):
    loss_meter = AverageMeter()
    running_loss = 0
    cnn.eval()
    rnn.eval()

    yhat = []
    y = []

    # for (i, (inputs, labels, lengths)) in enumerate(testloader):
    for (i, (inputs, labels, lengths)) in enumerate(tqdm.tqdm(testloader)):
        if inputs == None:
            continue
        # inputs = inputs.to(device)        
        labels = labels.to(device)
        lengths = lengths.to(device)

        # print("input shape: ", inputs.shape)
        # print("lengths: ", lengths)
        _, C, D, H, W = inputs.shape
        batch_size = lengths.shape[0]

        # optimizer.zero_grad()
        x_1 = torch.zeros_like(inputs)
            
        for idx, x1 in enumerate(inputs):
                x1 = get_image_patch_tensor_from_volume_batch(x1)
                x_1[idx] = data_augment(x1)

        inputs = x_1.to(device)
    
        with torch.set_grad_enabled(False):
            # with autocast():

            # inputs = reshape_images_cnn_input(inputs, lengths)
            # print("input of cnn shape ", inputs.shape)
            outputs = cnn(inputs)
            outputs = reshape_rnn_input(outputs, lengths, batch_size)
            # outputs.to(device)
            # print("input of rnn shape ",outputs.shape)
            outputs = rnn(outputs, lengths)
            # print(labels)
            # print()
            # print(outputs)
            loss = criterion(outputs.view(-1).float(), labels.float())


        outputs = outputs.to("cpu")

        loss_meter.update(loss.item())
        
        yhat.append(outputs.view(-1).detach().numpy())
        y.append(labels.to("cpu").numpy())

        # print("eval loss: ", loss.item())
        running_loss += loss.item()

    y = np.reshape(y, len(y) * batch_size)
    yhat = np.reshape(yhat, len(yhat) * batch_size)
    # print(y)
    # print(yhat)

    auc = sklearn.metrics.roc_auc_score(y, yhat)
    return loss_meter.average(), auc


def SimCLR(net, epoch, criterion, optimizer, trainloader, args):
    loss_meter = AverageMeter()
    running_loss = 0
    net.train()

    for (i, (b, _)) in enumerate(trainloader):

        optimizer.zero_grad()
        x_1 = torch.zeros_like(b).cuda()
        x_2 = torch.zeros_like(b).cuda()

        for idx, x in enumerate(b):
            x = get_image_patch_tensor_from_video_batch(x)

            x_1[idx] = data_augment(x)
            x_2[idx] = data_augment(x)


        out_1 = F.normalize(net(x_1), dim=1)
        out_2 = F.normalize(net(x_2), dim=1)
        
        indices = torch.arange(0, out_1.size(0), device=out_1.device)
        labels = torch.cat((indices, indices))

        loss = criterion(torch.cat([out_1, out_2], dim=0), labels)
        loss.backward()
        loss_meter.update(loss.item())
        optimizer.step()
        print(loss.item())
        running_loss += loss.item()

    return loss_meter.average(), running_loss

def checkpoint(net, model_store_folder, epoch_num, model, optimizer, lr_scheduler):
    print('Saving checkpoints...')
    
    checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch_num}


    suffix_latest = '{}_epoch_{}.pth'.format(model, epoch_num)
    # dict_net = net.state_dict()
    torch.save(checkpoint,
               '{}/{}'.format(model_store_folder, suffix_latest))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model related arguments
    parser.add_argument('--encoder_model', default='r3d_18')
    parser.add_argument('--encoder_pretrained', default=True)
    parser.add_argument('--supervised', default=True)
    parser.add_argument('--projection_size', default=64, type=int)
    parser.add_argument('--run', default=None, type=int, help='epoch to use weights')
    parser.add_argument('--checkpoint', default=None, type=int, help='epoch to start training from')
    # optimization related arguments
    parser.add_argument('--batch_size', default=2, type=int,
                        help='input batch size')
    parser.add_argument('--epoch', default=55, type=int,
                        help='epochs to train for')
    parser.add_argument('--dataset', default='oasis3')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', default=1e-3, type=float, help='LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--nesterov', default=False)
    parser.add_argument('--tau', default=0.1, type=float)
    # mode related arguments
    parser.add_argument('--eval', default=False)
    parser.add_argument('--output', default=None)

    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))
    
    if args.output is None:
        output_folder = "output"
    else:
        output_folder = args.output

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    pretrain_str = 'pretrained' if args.encoder_pretrained else 'random'
    model_store_folder = get_next_model_folder(\
                    'SimCLR_{}_{}'.format(args.encoder_model, pretrain_str), \
                    output_folder, args.run)
    try:
        os.mkdir(model_store_folder)
    except FileExistsError:
        print("Output folder exits")

    simclr_stats_csv_path = os.path.join(model_store_folder, "simclr_pred_stats.csv")
    regressor_stats_csv_path = os.path.join(model_store_folder, "regressor_pred_stats.csv")

    trainloader, testloader = dataloader(args.batch_size)
    

    if (args.supervised == True):
        cnn = construct_3d_enc(args.encoder_model, args.encoder_pretrained, \
            args.projection_size, 'supervised')
        rnn = construct_rnn()
        
        if device.type == "cuda":
            cnn = torch.nn.DataParallel(cnn)
            rnn = torch.nn.DataParallel(rnn)


        if (args.checkpoint != None):
            load_model(model_store_folder, args.checkpoint, cnn)
            load_model(model_store_folder, args.checkpoint, rnn)

        
        cnn = cnn.to(device)
        # inspect_model(cnn)
        rnn = rnn.to(device)
        # inspect_model(rnn)

        params = list(cnn.parameters()) + list(rnn.parameters())
        criterion = torch.nn.BCELoss()
        # optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2))
        # optimizer = optim.SGD(params, lr=args.lr, momentum=args.beta1, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
        
        print("\nStart supervised training!\n")
        scaler = GradScaler()
        bestLoss = float("inf")
        for epoch in range(1, args.epoch+1):
            train_loss, train_auc = train_supervise(cnn, rnn, epoch, criterion, optimizer, trainloader, scaler)
            print('epoch {} average train loss : {}'.format(epoch, train_loss))
            scheduler.step()
            #test(net, epoch, criterion, testloader, args)
            eval_loss, eval_auc = eval_supervise(cnn, rnn, epoch, criterion, testloader)
            print('epoch {} average eval loss : {}'.format(epoch, eval_loss))
            
            
            checkpoint(cnn, model_store_folder, epoch, "supervised_cnn", optimizer, scheduler)
            checkpoint(rnn, model_store_folder, epoch, "supervised_rnn", optimizer, scheduler)

            # Write stats into csv file
            stats = dict(
                    epoch      = epoch,
                    epoch_loss = train_loss,
                    train_auc = train_auc,
                    eval_loss = eval_loss,
                    eval_auc = eval_auc
                )
            write_csv_stats(simclr_stats_csv_path, stats)

            if eval_loss < bestLoss:
                # checkpoint(cnn, model_store_folder, epoch, "best_supervised_cnn", optimizer, scheduler)
                # checkpoint(rnn, model_store_folder, epoch, "best_supervised_rnn", optimizer, scheduler)
                bestLoss = eval_loss
        
        print("supervised training completed! Best loss: {}".format(bestLoss))
    




        



