import numpy as np
import cv2
import re
import os
from PIL import Image
import pandas as pd

from clipSegMask import initialzeClipSeg, computeClipSeg


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



inputDir = "dataset/training/frames"

pre_process_dir = "pre_processing"



# Load a model
model_yolo = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

processor, model_seg = initialzeClipSeg()
prompts = ["person","head", "hands", "feet", "torso"]



for path, subdirs, files in os.walk( os.path.join(".", inputDir), topdown=True):
            
    files.sort(key=natural_keys)


    if len(files) == 0:
         continue
    
    print(path)

    curr_preprocess = os.path.join(pre_process_dir,path.split("\\")[-1])

    if not os.path.exists(curr_preprocess):
        os.makedirs(curr_preprocess)

    if not os.path.exists(os.path.join(curr_preprocess,"face_obscure_imgs")):
        os.makedirs(os.path.join(curr_preprocess,"face_obscure_imgs"))

    if not os.path.exists(os.path.join(curr_preprocess,"pixelized_imgs")):
        os.makedirs(os.path.join(curr_preprocess,"pixelized_imgs"))

    if not os.path.exists(os.path.join(curr_preprocess,"blurred_imgs")):
        os.makedirs(os.path.join(curr_preprocess,"blurred_imgs"))

        

    roi_imgs = []
    roi_backgrounds = []
    open_pose_big_imgs = []
    roi_boxes = []
    full_images = []
    file_name = []
    roi_ids = []

    roi_img_path =[]
    roi_background_path = []
    roi_openpose_path = []

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
        augmented_img_obscure = np.asarray(curr_image).copy()
        augmented_img_blur = np.asarray(curr_image).copy()
        augmented_img_pixelate = np.asarray(curr_image).copy()


        # Init run get rois, backgrounds, open_poses

        count = 1
        for box, track_id in zip(boxes, track_ids):

            # if track_id ==3:
            x, y, w, h = box

            roi = img_arr[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)]

            # Segment and create rec background
            masks,_ = computeClipSeg(roi, prompts, processor, model_seg, 0.4)
            
            # blurred_mask = cv2.GaussianBlur(masks[0], (1, 1), 0)
            only_background = cv2.bitwise_and(roi,roi, mask= cv2.bitwise_not(masks[1]))

            only_background_body = cv2.bitwise_and(roi,roi, mask= cv2.bitwise_not(masks[0]))
            # only_foreground =  cv2.bitwise_and(roi,roi, mask= masks[0])

            min_kernel_size = 3
            kernel_size = max(min_kernel_size, int(max(x, y) * 0.05))

            # Ensure the kernel size is odd
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

        
            roi_blurred = cv2.GaussianBlur(roi, (kernel_size,kernel_size), 0)

            only_foreground_blurred =  cv2.bitwise_and(roi_blurred,roi_blurred, mask= masks[0])
            

            combined_blurred = only_foreground_blurred + only_background_body

            # Desired "pixelated" size
            wp, hp = (15, 15)
            height_roi, width_roi = roi.shape[:2]
            # Resize input to "pixelated" size
            temp = cv2.resize(roi, (wp, hp), interpolation=cv2.INTER_LINEAR)

            # Initialize output image
            roi_pixelated = cv2.resize(temp, (width_roi, height_roi), interpolation=cv2.INTER_NEAREST)

            only_foreground_pixelated =  cv2.bitwise_and(roi_pixelated,roi_pixelated, mask= masks[0])
            
            combined_pixelated = only_foreground_pixelated + only_background_body



            augmented_img_obscure[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)] = only_background
            augmented_img_blur[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)] = combined_blurred
            augmented_img_pixelate[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)] = combined_pixelated

            print(f"Finished box {count}\\{len(boxes)}")

            count+=1


            
        curr_file_name = file.split(".")[0]


        cv2.imwrite(os.path.join(curr_preprocess,"face_obscure_imgs", f"{curr_file_name}.jpg"), cv2.cvtColor(augmented_img_obscure, cv2.COLOR_BGR2RGB) )
        cv2.imwrite(os.path.join(curr_preprocess,"pixelized_imgs", f"{curr_file_name}.jpg"), cv2.cvtColor(augmented_img_blur, cv2.COLOR_BGR2RGB) )
        cv2.imwrite(os.path.join(curr_preprocess,"blurred_imgs", f"{curr_file_name}.jpg"), cv2.cvtColor(augmented_img_pixelate, cv2.COLOR_BGR2RGB) )

        print(f"Finished image {file}")
            
                 