import numpy as np
import cv2
import re
import os
from PIL import Image
import pandas as pd

from clipSegMask import initialzeClipSeg, computeClipSeg
from diffuserPoseGenerate import initializeDiffusers, calculateOpenPose, rescaleImage, resizeImage

import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator 

def atoi(text):
        return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
        '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



inputDir = "dataset\training\frames"

pre_process_dir = "\pre_processing"


rescale_image = 2.5


# Load a model YoloV8
model_yolo = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

processor, model_seg = initialzeClipSeg()
prompts = ["person","hands", "head", "feet", "torso"]


open_pose_model = initializeDiffusers()


background_img = np.asarray(Image.open("background.jpg"))




for path, subdirs, files in os.walk( os.path.join(".", inputDir), topdown=True):
            
    files.sort(key=natural_keys)

    if len(files) == 0:
         continue
    
    print(path)

    curr_preprocess = os.path.join(pre_process_dir,path.split("\\")[-1])

    if not os.path.exists(curr_preprocess):
        os.makedirs(curr_preprocess)

    if not os.path.exists(os.path.join(curr_preprocess,"roi_imgs")):
        os.makedirs(os.path.join(curr_preprocess,"roi_imgs"))

    if not os.path.exists(os.path.join(curr_preprocess,"roi_backgrounds")):
        os.makedirs(os.path.join(curr_preprocess,"roi_backgrounds"))

    if not os.path.exists(os.path.join(curr_preprocess,"roi_openpose")):
        os.makedirs(os.path.join(curr_preprocess,"roi_openpose"))

    if not os.path.exists(os.path.join(curr_preprocess,"roi_mask")):
        os.makedirs(os.path.join(curr_preprocess,"roi_mask"))

        

    roi_imgs = []
    roi_backgrounds = []
    open_pose_big_imgs = []
    roi_masks=[]
    roi_boxes = []
    full_images = []
    file_name = []
    roi_ids = []

    roi_img_path =[]
    roi_background_path = []
    roi_openpose_path = []
    roi_mask_path=[]

    open_pose_numBodies = []
    open_pose_totalScore = []
    open_pose_totalParts = []

    for file in files:
         
         
        curr_image = Image.open(os.path.join(path,file)) 
  
        results = model_yolo.track(curr_image, classes=0, persist=True)
        
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()

        img_arr = np.asarray(curr_image)
        augmented_img = np.asarray(curr_image).copy()



        # Init run get rois, backgrounds, open_poses
        for box, track_id in zip(boxes, track_ids):

            x, y, w, h = box

            roi = img_arr[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)]

            # Segment and create rec background
            masks,_ = computeClipSeg(roi, prompts, processor, model_seg, 0.4)
            
            # blurred_mask = cv2.GaussianBlur(masks[0], (1, 1), 0)
            only_background = cv2.bitwise_and(roi,roi, mask= cv2.bitwise_not(masks[0]))


            roi_background = background_img[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)]
            only_foreground =  cv2.bitwise_and(roi_background,roi_background, mask= masks[0])
            rec_background = only_background + only_foreground
        
            # Extract pose and scale it up
            open_pose_img, curr_pose = calculateOpenPose(roi,open_pose_model,256, 64)
            open_pose_img_big = rescaleImage(open_pose_img, rescale_image)


            roi_imgs.append(roi)
            roi_backgrounds.append(rec_background)
            open_pose_big_imgs.append(open_pose_img_big)
            roi_boxes.append(box)
            roi_ids.append(track_id)

            roi_masks.append(masks[0])
            
            full_images.append(augmented_img)
            file_name.append(file)

            curr_file_name = file.split(".")[0]



            
                 
            cv2.imwrite(os.path.join(curr_preprocess,"roi_imgs", f"{curr_file_name}_{track_id}.jpg"), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) )

            cv2.imwrite(os.path.join(curr_preprocess,"roi_backgrounds", f"{curr_file_name}_{track_id}.jpg"), cv2.cvtColor(rec_background, cv2.COLOR_BGR2RGB) )

            cv2.imwrite(os.path.join(curr_preprocess,"roi_mask", f"{curr_file_name}_{track_id}.jpg"),(masks[0]).astype(np.uint8) )
                 
            cv2.imwrite(os.path.join(curr_preprocess,"roi_openpose", f"{curr_file_name}_{track_id}.jpg"), cv2.cvtColor(open_pose_img_big, cv2.COLOR_BGR2RGB) )

            roi_img_path.append(os.path.join(curr_preprocess,"roi_imgs", f"{curr_file_name}_{track_id}.jpg"))
            roi_background_path.append(os.path.join(curr_preprocess,"roi_backgrounds", f"{curr_file_name}_{track_id}.jpg"))
            roi_openpose_path.append(os.path.join(curr_preprocess,"roi_openpose", f"{curr_file_name}_{track_id}.jpg"))

            roi_mask_path.append(os.path.join(curr_preprocess,"roi_mask", f"{curr_file_name}_{track_id}.jpg"))

            open_pose_numBodies.append(len(curr_pose))
            
            if len(curr_pose) > 0:
                open_pose_totalScore.append(curr_pose[0].body.total_score)
                open_pose_totalParts.append(curr_pose[0].body.total_parts)
            elif len(curr_pose) == 0:
                open_pose_totalScore.append(-1)
                open_pose_totalParts.append(-1) 


            print(f"Extract info from image {file} {track_id}")


    percentile_list = pd.DataFrame(np.column_stack([pd.Series(file_name), pd.Series(roi_ids), pd.Series(roi_boxes), pd.Series(roi_img_path), pd.Series(roi_background_path), pd.Series(roi_openpose_path), pd.Series(open_pose_numBodies), pd.Series(open_pose_totalScore), pd.Series(open_pose_totalParts), pd.Series(roi_mask_path)]), 
                                columns=['file_name', 'roi_ids', 'roi_boxes', 'roi_img_path', 'roi_background_path', 'roi_openpose_path', 'open_pose_numBodies','open_pose_totalScore', 'open_pose_totalParts', 'roi_mask_path'])

    print(percentile_list)
    percentile_list.to_csv(os.path.join(curr_preprocess,"pre_process_csv.csv"), encoding='utf-8', index=False)

