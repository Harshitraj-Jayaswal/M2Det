from __future__ import print_function
import os
import warnings
warnings.filterwarnings('ignore')
import torch
import pickle
import argparse
import numpy as np
from m2det import build_net
from utils.timer import Timer
import torch.backends.cudnn as cudnn
from layers.functions import Detect,PriorBox
from data import BaseTransform
from configs.CC import Config
from tqdm import tqdm
from utils.core import *

parser = argparse.ArgumentParser(description='M2Det Testing')
parser.add_argument('-c', '--config', default='configs/m2det320_vgg.py', type=str)
parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default=None, type=str, help='Trained state_dict file path to open')
parser.add_argument('--test', action='store_true', help='to submit a test file')
args = parser.parse_args()

print_info('----------------------------------------------------------------------\n'
           '|                       M2Det Evaluation Program                     |\n'
           '----------------------------------------------------------------------', ['yellow','bold'])
global cfg
cfg = Config.fromfile(args.config)
if not os.path.exists(cfg.test_cfg.save_folder):
    os.mkdir(cfg.test_cfg.save_folder) #Making eval folder
anchor_config = anchors(cfg) #Anchor configurations
print_info('The Anchor info: \n{}'.format(anchor_config))
priorbox = PriorBox(anchor_config)
with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        priors = priors.cuda()

def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder) #making eval/COCO folder if not made

    num_images = len(testset) #Number of images
    print_info('=> Total {} images to test.'.format(num_images),['yellow','bold'])
    num_classes = cfg.model.m2det_config.num_classes #number of classes
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)] # a list of size (num_classes,num_images) where each element is an empty list->[]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl') #det_file=eval/COCO/detection.pkl
    tot_detect_time, tot_nms_time = 0, 0
    print_info('Begin to evaluate',['yellow','bold'])
    for i in tqdm(range(num_images)):
        img = testset.pull_image(i) #taking one image
        # step1: CNN detection
        _t['im_detect'].tic() #starting the time

        boxes, scores = image_forward(img, net, cuda, priors, detector, transform) 
        '''Making prediction which includes the process: 
        image preprocessing->inputing image to model->getting predictions->passing through detector() function [which decodes the location preds(boundry box coordinates) and apply nms]->getting o/p i.e. boxes and scores'''

        detect_time = _t['im_detect'].toc() #return total time for prediction for one image
         
        # step2: Post-process: NMS
        _t['misc'].tic() #Starting the timer
        nms_process(num_classes, i, scores, boxes, cfg, thresh, all_boxes, max_per_image) #applying nms for that one image (after nms all the coordinates and scores in all_boxes for every image after the loop ends)
        nms_time = _t['misc'].toc() #return the total time for nms for that one image

        tot_detect_time += detect_time if i > 0 else 0
        tot_nms_time += nms_time if i > 0 else 0

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL) #saving the predictions
    print_info('===> Evaluating detections',['yellow','bold'])
    testset.evaluate_detections(all_boxes, save_folder)
    print_info('Detect time per image: {:.3f}s'.format(tot_detect_time / (num_images-1)))
    print_info('Nms time per image: {:.3f}s'.format(tot_nms_time / (num_images - 1)))
    print_info('Total time per image: {:.3f}s'.format((tot_detect_time + tot_nms_time) / (num_images - 1)))
    print_info('FPS: {:.3f} fps'.format((num_images - 1) / (tot_detect_time + tot_nms_time)))

if __name__ == '__main__':
    net = build_net('test',
                    size = cfg.model.input_size,
                    config = cfg.model.m2det_config) #Building the model
    init_net(net, cfg, args.trained_model) #Loading the pretrained weights for the model       
    print_info('===> Finished constructing and loading model',['yellow','bold'])

    net.eval() #Mentioning model is for evaluation
           
    _set = 'eval_sets' if not args.test else 'test_sets'
    testset = get_dataloader(cfg, args.dataset, _set) #Loading the test dataset with the help of class
           
    if cfg.test_cfg.cuda:
        net = net.cuda() #network will load in gpu
        cudnn.benchmark = True
    else:
        net = net.cpu()

    detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)
    '''Decode location preds, apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations. Initailizing the Detect() function here'''

    save_folder = os.path.join(cfg.test_cfg.save_folder, args.dataset) #save_folder=eval/COCO
    _preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1)) #Initializing BaseTransorm() function here for image preporcessing
    test_net(save_folder, 
             net, 
             detector, 
             cfg.test_cfg.cuda, 
             testset, 
             transform = _preprocess, 
             max_per_image = cfg.test_cfg.topk, 
             thresh = cfg.test_cfg.score_threshold)
