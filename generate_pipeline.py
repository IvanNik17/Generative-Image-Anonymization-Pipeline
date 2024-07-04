import numpy as np
import cv2
import re
import os
from PIL import Image
import pandas as pd

from clipSegMask import initialzeClipSeg, computeClipSeg
from diffuserPoseGenerate import initializeDiffusers, calculateOpenPose, rescaleImage, resizeImage
from diffuserAnimationGen import initializeDiffusersAnimate, generateAnimate

import torch

from prompt_randomizer import prompt_randomize

from ast import literal_eval
import shlex

def match_saturation_hist(image1, image2):

    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    hist1, _ = np.histogram(hsv1[:,:,1], bins=256, range=[0,256])
    hist2, _ = np.histogram(hsv2[:,:,1], bins=256, range=[0,256])


    cdf1 = hist1.cumsum()
    cdf2 = hist2.cumsum()
    cdf1 = cdf1 / cdf1[-1]
    cdf2 = cdf2 / cdf2[-1]

    lut = np.interp(cdf2, cdf1, np.arange(256))
    hsv2[:,:,1] = np.clip(lut[hsv2[:,:,1]], 0, 255).astype(np.uint8)

    matched_image2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    return matched_image2

def atoi(text):
        return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
        '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


sequence = '01'

 

result_dir = "\results"

inputDir = "dataset\training\frames"

pre_process_dir = "\pre_processing"

output_gens = "\output_gens"

rescale_image = 2.5


processor, model_seg = initialzeClipSeg()
prompts = ["person","hands", "head", "feet", "torso"]


# open_pose_model, diffuser_pipe = initializeDiffusers()

open_pose_model = initializeDiffusers()

diffuser_animate_pipe = initializeDiffusersAnimate()


negative_prompt = """anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, colorful background, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, blurry, bad anatomy, bad proportions, extra hands, saturated colors, Disfigured, bad art, amateur, poorly drawn, ugly, flat, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits , cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry
# ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), (( poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs )), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), sexual"""




pre_process_df_big = pd.read_csv(os.path.join(pre_process_dir, sequence, "pre_process_csv.csv"))

pre_process_df_big['numeric_filename'] = pre_process_df_big['file_name'].str.extract(r'(\d+)').astype(int)

df_st_30 = pre_process_df_big[pre_process_df_big['numeric_filename'] < 30]
df_b_30_and_60 = pre_process_df_big[(pre_process_df_big['numeric_filename'] >= 30) & (pre_process_df_big['numeric_filename'] < 60)]
df_b_60_and_90 = pre_process_df_big[(pre_process_df_big['numeric_filename'] >= 60) & (pre_process_df_big['numeric_filename'] < 90)]



for curr_df in [df_st_30, df_b_30_and_60, df_b_60_and_90]:
    pre_process_df = []
    pre_process_df = curr_df

    all_big_images = []
    all_big_image_names = []
    for img_p in pre_process_df.file_name.unique():
        base_path = os.path.join(".", inputDir, sequence, img_p)
        all_big_images.append(np.array(Image.open(base_path)))
        all_big_image_names.append(img_p)

    big_images_dict= dict(zip(all_big_image_names, all_big_images)) 
    
    print(len(all_big_images))
                
    print(pre_process_df)

    all_track_ids = pre_process_df.roi_ids.unique()

    for i in all_track_ids:

        id_curr = i


        positive_prompt = prompt_randomize()
        print(positive_prompt)

        curr_track_id_df = pre_process_df.loc[pre_process_df['roi_ids'] == id_curr]

        print(curr_track_id_df)


        all_open_pose = []

        num_rows = curr_track_id_df.shape[0]
        tresh_no_body = 0.2
        tresh_no_score = 0.25
        print(num_rows)

        print(f"count total score less {(curr_track_id_df['open_pose_totalScore'].lt(10).sum() - curr_track_id_df['open_pose_totalScore'].eq(-1).sum())}")

        print(f"count total parts less {(curr_track_id_df['open_pose_totalParts'].lt(10).sum() - curr_track_id_df['open_pose_totalParts'].eq(-1).sum())}")
        print(curr_track_id_df['open_pose_numBodies'].eq(0).sum())



        if curr_track_id_df['open_pose_numBodies'].eq(0).sum() > num_rows* tresh_no_body or ( (curr_track_id_df['open_pose_totalScore'].lt(10).sum() - curr_track_id_df['open_pose_totalScore'].eq(-1).sum()) > num_rows * tresh_no_score and  (curr_track_id_df['open_pose_totalParts'].lt(10).sum() - curr_track_id_df['open_pose_totalParts'].eq(-1).sum()) > num_rows * tresh_no_score):
            print(f"current ID {id_curr} will be skipped!!!")
            continue
        

        for index, row in curr_track_id_df.iterrows():
            
            all_open_pose.append(Image.open(row["roi_openpose_path"]))
        



        # Generate new synthetic sub-images
        gen_results = generateAnimate(diffuser_animate_pipe,all_open_pose,positive_prompt,negative_prompt)

        # Augment the images
        counter = 0
        for index, row in curr_track_id_df.iterrows():

            row["roi_boxes"] = row["roi_boxes"].strip('[]')
            row["roi_boxes"] = shlex.split(row["roi_boxes"])

            x, y, w, h = row["roi_boxes"]
            x = int(float(x))
            y = int(float(y))
            w = int(float(w))
            h = int(float(h))
            
            gen_image = gen_results[counter]

            # Segment syntheic make mask
            masks_gen, mask_gen_soft = computeClipSeg(np.array(gen_image), prompts, processor, model_seg, 0.3)
            
            curr_img_roi =  np.array(Image.open(row["roi_img_path"]))
            curr_background_roi =  np.array(Image.open(row["roi_background_path"]))
            
            # Scale down syntheic
            gen_image_initSize = resizeImage(np.array(gen_image), (curr_img_roi.shape[1],curr_img_roi.shape[0]))

            # Scale down mask
            gen_mask_initSize = resizeImage(np.array(masks_gen[0]), (curr_img_roi.shape[1],curr_img_roi.shape[0]))


            gen_mask_initSize = gen_mask_initSize.astype('float')/255
            blurred_mask = cv2.GaussianBlur(gen_mask_initSize, (5, 5), 0)
            mask_blurred_3chan = np.repeat(blurred_mask[:, :, np.newaxis], 3, axis=2)

            big_curr = big_images_dict[row["file_name"]]
            roi_curr = big_curr[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)]

            img_gen = gen_image_initSize.astype('float') / 255.
            img_gen = cv2.blur(img_gen, (2,2))
            bg = curr_background_roi.astype('float') / 255.
            out  = bg * (1 - mask_blurred_3chan) + img_gen * mask_blurred_3chan
            combined = (out * 255).astype('uint8')
            
            test_name = row["file_name"]
            cv2.imwrite(os.path.join(output_gens,f"{counter}_{test_name}"), cv2.cvtColor(combined, cv2.COLOR_BGR2RGB) )   
            cv2.imwrite(os.path.join(output_gens,f"{counter}_bla_{test_name}"), cv2.cvtColor(gen_image_initSize, cv2.COLOR_BGR2RGB) ) 

            # Add synthetic image to larger image
            augmented_img = big_images_dict[row["file_name"]]

            if augmented_img[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)].shape == combined.shape:


                augmented_img[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)] = combined
            else:
                w_space, h_space = augmented_img[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)].shape[0:2]
                combined = cv2.resize(combined, (h_space, w_space))
                augmented_img[int(y)-int(h/2):int(y)+int(h/2), int(x)-int(w/2):int(x)+int(w/2)] = combined

            big_images_dict[row["file_name"]] = augmented_img

            show_name = row["file_name"]
            show_id = row["roi_ids"]
            print(f"Augmented with synthetic image {show_name} {show_id}")

            counter+=1

        print(f"Current ID {id_curr} images Finished")

    if not os.path.exists(os.path.join(result_dir,sequence)):
        os.makedirs(os.path.join(result_dir,sequence))         
            
    for index, row in pre_process_df.iterrows():
                
        augmented_img = big_images_dict[row["file_name"]]
        show_name = row["file_name"]
        show_id = row["roi_ids"] 

        cv2.imwrite(os.path.join(result_dir,sequence,show_name), cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB) )    

        print(f"Saved  synthetic image {show_name} {show_id}")

             
    torch.cuda.empty_cache()