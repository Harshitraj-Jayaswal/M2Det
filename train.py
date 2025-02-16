from __future__ import print_function
import os  #Provides functions for interacting with the Operating System. Functions like interactiong with file system.
import warnings 
warnings.filterwarnings('ignore')

import time
import torch
import shutil
import argparse '''Helps to create a program in a command line environment in a way that appears not only easy to use but also improves 
interaction with the help of arguments passed during the calling of the program.'''
from m2det import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from data import detection_collate
from configs.CC import Config
from utils.core import *

#Creating the argeparse object that will store all the necessary information that has to be passed from python command line
parser = argparse.ArgumentParser(description='M2Det Training')

#Adding the arguments for the Parser object, meaning, how to take arguments from the command line
parser.add_argument('-c', '--config', default='configs/m2det320_vgg16.py')
parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO dataset')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool, default=False, help='Use tensorborad to show the Loss Graph')

args = parser.parse_args() 
'''The information gathered above is stored and used when arguments are parsed through parse_args().'''

print_info('----------------------------------------------------------------------\n'
           '|                       M2Det Training Program                       |\n'
           '----------------------------------------------------------------------',['yellow','bold'])

logger = set_logger(args.tensorboard)
global cfg
cfg = Config.fromfile(args.config) #Gives file name and nueral network configuration in the form of dictionary

net = build_net('train', 
                size = cfg.model.input_size, # Only 320, 512, 704 and 800 are supported
                config = cfg.model.m2det_config) #Buiding the M2Det network with configurations stored in cfg. net is the object of M2det class

init_net(net, cfg, args.resume_net) # init the network with pretrained weights or resumed weights

if args.ngpu>1:
    net = torch.nn.DataParallel(net)
if cfg.train_cfg.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = set_optimizer(net, cfg) #initializing optimizer for weight updates and using stochastic gradient descent
criterion = set_criterion(cfg) #initializing criterion for Multibox loss calculation
priorbox = PriorBox(anchors(cfg)) #initilaizing priorbox to Compute priorbox coordinates in center-offset form for each source feature map.

with torch.no_grad(): #under torch.no_grad, all the things are not saved to use it for gradient calculation
    priors = priorbox.forward()
    if cfg.train_cfg.cuda:
        priors = priors.cuda()

if __name__ == '__main__':
    net.train() #model.train() tells your model that you are training the model (Here, model is net)
    epoch = args.resume_epoch
    print_info('===> Loading Dataset...',['yellow','bold'])
    dataset = get_dataloader(cfg, args.dataset, 'train_sets')
    epoch_size = len(dataset) // (cfg.train_cfg.per_batch_size * args.ngpu)
    max_iter = getattr(cfg.train_cfg.step_lr,args.dataset)[-1] * epoch_size
    stepvalues = [_*epoch_size for _ in getattr(cfg.train_cfg.step_lr, args.dataset)[:-1]]
    print_info('===> Training M2Det on ' + args.dataset, ['yellow','bold'])
    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset, 
                                                  cfg.train_cfg.per_batch_size * args.ngpu, 
                                                  shuffle=True, 
                                                  num_workers=cfg.train_cfg.num_workers, 
                                                  collate_fn=detection_collate))
            if epoch % cfg.model.save_eposhs == 0:
                save_checkpoint(net, cfg, final=False, datasetname = args.dataset, epoch=epoch)
            epoch += 1
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, cfg.train_cfg.gamma, epoch, step_index, iteration, epoch_size, cfg)
        images, targets = next(batch_iterator)
        if cfg.train_cfg.cuda:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        out = net(images)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        write_logger({'loc_loss':loss_l.item(),
                      'conf_loss':loss_c.item(),
                      'loss':loss.item()},logger,iteration,status=args.tensorboard)
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        print_train_log(iteration, cfg.train_cfg.print_epochs,
                            [time.ctime(),epoch,iteration%epoch_size,epoch_size,iteration,loss_l.item(),loss_c.item(),load_t1-load_t0,lr])
    save_checkpoint(net, cfg, final=True, datasetname=args.dataset,epoch=-1)
