import sys, os
import time
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

from torchvision import transforms
from torchvision.io import read_image, ImageReadMode


from dataloader import get_data
# from network import ReconstructionMLP, AutoEncoderNet

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
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *
from sklearn.metrics import roc_auc_score, precision_score, recall_score
# from skimage import measure
from utils import *
import random
import glob
import sklearn

import argparse

def point_score(outputs, imgs):
    
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score

def read_data(data_dir, humans_only):

    test_data = get_data(data_dir, batch_size=1, humans_only=humans_only, augment=False, shuffle=False)
    # test_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and not f.endswith("csv")]

    annotations_file = os.path.join(data_dir,"target_labels.csv")
    image_data = pd.read_csv(annotations_file) # Video  Track  Class  NumFrames  NumFeatures

    test_files = []

    for row in image_data.iterrows():
        filename = "{0}_{1}.png".format(str(row[1].Video).zfill(2), str(row[1].Track).zfill(6))
        assert os.path.exists(os.path.join(data_dir,filename)), "File '{0}' does not exist.".format(filename)
        
        test_files.append(filename)

    if humans_only == True:
        class_arr = np.array(image_data.get('Class').values, dtype=np.int32)
        target_args = np.where(class_arr == 0)[0]

        test_files = list(np.array(test_files)[target_args])
        image_data = image_data.iloc[target_args]

        assert len(test_files) == len(image_data)
    


    data_dict = {}
    data_dict["test_data"] = test_data
    data_dict["image_data"] = image_data
    data_dict["test_files"] = test_files


    return data_dict


# def read_model(save_dir):
    
#     model = AutoEncoderNet(dropout_hidden=0.0, dropout_input=0.0)
#     # model = ReconstructionMLP(8192)

#     model.load_state_dict(torch.load(save_dir))
#     model.eval()


#     return model


def evaluate_model(raw_eval, data_dir, data_dict, model, m_items):

    visualize = False

    test_data = data_dict.get('test_data')
    test_files = data_dict.get('test_files')
    image_data = data_dict.get('image_data')

    mse = torch.nn.MSELoss()
    errors = []

    

    with torch.no_grad():

        for X, _, _ in tqdm(test_data):

            
            X = X.repeat(1, 3, 1, 1)
            X = X.to("cuda")
            # latent_vector, y_pred, mu, logvar  = model(X)

            y_pred, feas, updated_feas, m_items, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(X, m_items, False)
            
            error = mse(y_pred, X)
            errors.append(error.cpu())

            point_sc = point_score(y_pred, X)

            if  point_sc < 0.01:
                query = F.normalize(feas, dim=1)
                query = query.permute(0,2,3,1) # b X h X w X d
                m_items_dir = model.memory.update(query, m_items, False)
        
        
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        anomaly_thresh = avg_error + std_error
        anomalous_track_args = np.where(errors > anomaly_thresh)[0]
        print("Detected {0} anomalous tracks.".format(len(anomalous_track_args)), flush=True)


        frame_scores_dict = {}
        a_id = -1

        for X, _, _ in tqdm(test_data):
            
            a_id += 1

            # if a_id in anomalous_track_args:
                
            filename = test_files[a_id]
            file_record = image_data.iloc[a_id]

            video_name = str(int(file_record.get('Video'))).zfill(2)
            track_id = int(file_record.get('Track'))
            obj_class = int(file_record.get('Class'))
            boxes = [ np.array(x.split(', '), dtype=np.float32) for x in list(file_record.get('Boxes')[2:-2].split("], [")) ]
            filenames = list(file_record.get('Filenames')[1:-1].replace("'", "").split(", "))
            
            num_frames = int(file_record.get('NumFrames'))

            
            assert len(filenames) == num_frames
            assert str(filename.split('_')[0]) == video_name
            assert int(filename.split('_')[1].split('.')[0]) == track_id

            X_orig = torch.tensor(np.array(read_image(os.path.join(data_dir,filename), mode=ImageReadMode.GRAY), dtype=np.float32) / 255.0)

            assert X_orig.shape[1] == num_frames

            # y_pred_ = model(X)
            X = X.repeat(1, 3, 1, 1)
            X = X.to("cuda")
            # latent_vector, y_pred_, mu, logvar  = model(X)
            y_pred_, feas, updated_feas, m_items, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(X, m_items, False)

            point_sc = point_score(y_pred_, X)

            if  point_sc < 0.01:
                query = F.normalize(feas, dim=1)
                query = query.permute(0,2,3,1) # b X h X w X d
                m_items_dir = model.memory.update(query, m_items, False)

            resize_to = (X_orig.shape[1], X_orig.shape[2])
            y_pred = transforms.functional.resize(y_pred_, resize_to, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            y_pred = y_pred.cpu()
            frame_wise_mse = []

            

            for frame_id in range(num_frames):
                pred_shape = y_pred[0, 0, frame_id, :]
                true_shape = X_orig[0, frame_id, :]

                shape_error = mse(pred_shape, true_shape)
                frame_wise_mse.append(np.float32(shape_error))


            fwise_mse_smooth__std_3 = gaussian_filter1d(frame_wise_mse, 3)
            fwise_mse_smooth__std_6 = gaussian_filter1d(frame_wise_mse, 6)

            if raw_eval:
                save_data = frame_wise_mse.copy()
            else:
                save_data = np.copy(fwise_mse_smooth__std_6)
            

            for id, score in enumerate(save_data):
                file = filenames[id]
                bbox = boxes[id]

                prev_results = frame_scores_dict.get(file)
                if prev_results is None:
                    frame_scores_dict[file] = [1, [score], [obj_class], [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]], [1.0], [track_id]]
                else:
                    prev_results[0] += 1
                    prev_results[1].append(score)
                    prev_results[2].append(obj_class)
                    prev_results[3].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                    prev_results[4].append(1.0)
                    prev_results[5].append(track_id)
                    frame_scores_dict.update({ file: prev_results })



            if visualize:
                fig = plt.figure()
                _, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True)
                # ax1 = plt.subplot(1,3,1)
                # ax1 = plt.subplot2grid((1, 3), (0, 0))
                ax1.matshow(X_orig[0,:,:])
                ax1.set_title("Original Image")
                ax1.set_axis_off()
                ax1.set_ylim([0, num_frames])
                
                # ax2 = plt.subplot(1,3,2, sharey=ax1)
                # ax2 = plt.subplot2grid((1, 3), (0, 1))
                ax2.matshow(y_pred[0,0,:,:])
                ax2.set_title("Reconstructed Image")
                ax2.set_axis_off()
                ax2.set_ylim([0, num_frames])

                # plt.subplot(1,3,3, sharey=ax2)
                # ax3 = plt.subplot2grid((1, 3), (0, 2))
                ax3.plot(fwise_mse_smooth__std_3, np.arange(num_frames), '-', linewidth=2, color='#33cccc', label="sigma = 3")
                ax3.plot(fwise_mse_smooth__std_6, np.arange(num_frames), '-', linewidth=2, color='#ffcc66', label="sigma = 6")
                ax3.plot(frame_wise_mse, np.arange(num_frames), '.', markersize=4, color='#cc99ff', label="original")
                ax3.set_ylabel("Frame", fontsize=14)
                # ax3.set_yticks([])
                ax3.set_xlabel("MSE", fontsize=14)
                ax3.legend()
                ax3.set_ylim([0, num_frames])
                
                plt.suptitle("Shape Reconstruction Error:\nFile {0}".format(filename), fontweight='bold', fontsize=18)
                plt.savefig("./test.png")
                
                plt.show()
                fig.clf()
                plt.close()

                time.sleep(3)

    


    return frame_scores_dict
   




def main(humans_only, raw_eval, data_load_dir, model_load_dir, save_results_dir):

    data_dict = read_data(data_load_dir, humans_only)
    # model = read_model(model_load_dir)

    state_dict_path = r"C:\Work\2022_workstuff\Research\anomalyDetectors\MNAD\log_dir\avenue_traces\model_59.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(state_dict_path)

    m_items_dir = r"C:\Work\2022_workstuff\Research\anomalyDetectors\MNAD\log_dir\avenue_traces\keys_59.pt"
    m_items = torch.load(m_items_dir)
    model.cuda()
    # model = VAE_CLEAN(in_channels=3, encoder_shape=128, decoder_shape=128, latent_dim=2000)
    # model = model.to(device)

    # model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    frame_scores_dict = evaluate_model(raw_eval, data_load_dir, data_dict, model, m_items)


    # Save one file per video
    save_results_dir += "/test-frame-all-object-scores--subset-{0}.txt"
    prev_video, save_dir = None, None

    for item in frame_scores_dict.items():
        
        curr_video = item[0].split('_')[0]

        

        if (prev_video is None) or (prev_video != curr_video):
            prev_video = curr_video
            save_dir = save_results_dir.format(curr_video)
        
        with open(save_dir, 'a') as f:
            curr_list = item[1][1]
            # curr_list = curr_list.sort()
            # f.write(str(curr_list) + '\n')
            f.write(str(item) + '\n')

    print("Saved results in file {0}.".format(save_dir), flush=True)
    
    

    return 0




if __name__ == "__main__":

    
    datasets = [ "Avenue" ]
    humans_only = True

    # Setting this flag to False results in applying 1D Gaussian smoothing
    # to the results with sigma = 6 (kernel size calculated automatically)
    raw_eval = True



    data_load_dir = r"C:\Work\2024_workstuff\Python\Variational_Autoencoder_Generate_Synthetic_Surveillance_Data-main\images\test"

    root_save_dir = r"C:\Work\2024_workstuff\Python\Variational_Autoencoder_Generate_Synthetic_Surveillance_Data-main"
    # root_save_dir = "./output/models/{0}/2024-04-16/09:54:07/"
    model_load_dir = root_save_dir + "/model__state_dict.pt"
    save_results_dir = root_save_dir + "/results_MNAD/"

    if raw_eval:
        save_results_dir += "raw/"
    else:
        save_results_dir += "smoothed/"
        


    for id, dataset_name in enumerate(datasets):
        save_dir = save_results_dir #.format(dataset_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        main(humans_only, raw_eval, data_load_dir, model_load_dir, save_dir)
        


    print("\n\n")
    print("Done.")
    print("\n\n\n")
    sys.exit()