import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
#import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.lgn_net import *
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import sklearn
from utils import *
import random
import glob

import argparse
import scipy.io as scio

from skimage import measure



parser = argparse.ArgumentParser(description="LGN-Net")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--lambdas', type=float, default=0.6, help='weight for the anomality score')   
parser.add_argument('--gamma', type=float, default=0.009, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--model_dir', type=str,  default=r"C:\Work\2022_workstuff\Research\anomalyDetectors\LGN-Net\log_dir\avenue_small_obscured\60model.pth",help='directory of model')
parser.add_argument('--m_items_dir', type=str, default=r"C:\Work\2022_workstuff\Research\anomalyDetectors\LGN-Net\log_dir\avenue_small_obscured\60keys.pt",help='directory of model')


if __name__ == "__main__":
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = "0"
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    # test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"
    test_folder = r"C:\Work\2022_workstuff\Research\anomalyDetectors\MNAD\dataset"+"/"+args.dataset_type+"/testing/frames"

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
                transforms.ToTensor(),            
                ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_size = len(test_dataset)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model


    model = lgn(n_channel =3,  t_length = 5, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1)

    model.load_state_dict(torch.load(args.model_dir))
    model.cuda()
    m_items = torch.load(args.m_items_dir)
    # labels = np.load('./labels/frame_labels_'+args.dataset_type+'.npy')
    labels = np.load('./labels/frame_labels_avenue.npy')
    # labels = [labels]
    print(labels)

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    
    for i in range(0, len(videos_list)):
        videos_list[i] = videos_list[i].replace("\\", "/" )


    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])
    
    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        #The first four frames are unpredictable
        labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    
    print(labels_list.shape)
    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    m_items_test = m_items.clone()
    model.eval()


    for k,(imgs) in enumerate(test_batch):
        

        if k == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']


        imgs = Variable(imgs).cuda()

        # main_save_dir = r"D:\2022_workstuff\Research\anomalyDetectors\LGN-Net\result_images"
            
        parts_type = videos_list[video_num].split('/')[-3]
        parts_num_dir = videos_list[video_num].split('/')[-1]

        # save_dir = os.path.join(main_save_dir, f"{args.dataset_type}",f"{parts_type}","frames", f"{parts_num_dir}")

        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        

        outputs, feas, updated_feas,updated_orig, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:12], m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,12:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        inp = (imgs[0,3*4:]+1)/2
        rec = (outputs[0] + 1) / 2
        inp_img = inp.detach().cpu().numpy()[0]
        rec_img = rec.detach().cpu().numpy()[0]

        # score,full_img = measure.compare_ssim(inp_img, rec_img, full=True)
        # # print(full_img)


        # saved_img = (full_img * 255).astype(np.uint8)

        # inp_image = (inp_img * 255).astype(np.uint8)
        # rec_image = (rec_img * 255).astype(np.uint8)
        
        # cv2.imwrite(os.path.join(save_dir, f"{k}_in.jpg"),inp_image)
        # cv2.imwrite(os.path.join(save_dir, f"{k}_out.jpg"),rec_image)
        # cv2.imwrite(os.path.join(save_dir, f"{k}_error.jpg"),saved_img)
        # print(f"saved image {k} in {save_dir}")

        # print(f"Finished image {k}")
        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,12:])
        


        if  point_sc < args.gamma:
            query = F.normalize(feas, dim=1)
            query = query.permute(0,2,3,1) # b X h X w X d
            m_items_test = model.memory.update(query, m_items_test, False)

        psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
        feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)

    # Measuring the abnormality score and the AUC
    all_gt = []
    anomaly_score_total_list = []

    print(args.lambdas)
    for video in sorted(videos_list):

        video_name = video.split('/')[-1]
        
        anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                        anomaly_score_list_inv(feature_distance_list[video_name]), args.lambdas)

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    # np.savetxt('anom_score.txt', anomaly_score_total_list, delimiter=',')

    # np.savetxt('anom_labels.txt', labels_list, delimiter=',')

    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

    print('The result of ', args.dataset_type)
    print('AUC: ', accuracy*100, '%')

    true_labels =np.squeeze( np.expand_dims(1-labels_list, 0), axis=0)
    anom_score = np.squeeze(anomaly_score_total_list)
    anom_score_pred = np.where(anom_score > 0.5, 1, 0)

    sensitivity = sklearn.metrics.recall_score(true_labels , anom_score_pred)
    specificity = sklearn.metrics.recall_score(np.logical_not(true_labels) , np.logical_not(anom_score_pred))
    print(f"sensitivity {sensitivity}")
    print(f"specificity {specificity}")

    precision = precision_score(true_labels , anom_score_pred)
    recall = recall_score(true_labels , anom_score_pred)

    print(f"precision {precision}")
    print(f"recall {recall}")


    # mean psnr
    a=-1
    c=0
    n_nor = n_abnor = 0
    avg_nor = avg_abnor = 0
    for video in sorted(videos_list):
        a+=1
        video_name = video.split('/')[-1]
        for  b in range(0, len(psnr_list[video_name])):
            
            if labels_list[c] == 0:
                n_nor += 1
                avg_nor += psnr_list[video_name][b]
            else:
                n_abnor += 1
                avg_abnor += psnr_list[video_name][b]
            c+=1

    print("abnor_psnr==",avg_abnor/n_abnor,"nor_psnr==",avg_nor/n_nor)



