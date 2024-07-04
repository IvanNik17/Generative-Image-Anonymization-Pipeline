from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import numpy as np
import cv2
from torch import nn

def initialzeClipSeg():
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    return processor, model

def computeClipSeg(image, prompts, processor, model, threshold = 0.4):


    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
    # predict
    with torch.no_grad():
        outputs = model(**inputs)

    preds = nn.functional.interpolate(
        outputs.logits.unsqueeze(1),
        size=(image.shape[0], image.shape[1]),
        mode="bilinear"
    )

    masks = []
    smooth_masks = []

    for i in range(len(prompts)):

        curr_pred = preds[i][0]
        mask_2 = torch.sigmoid(curr_pred).cpu().numpy()


        pred_mask = torch.sigmoid(curr_pred).cpu().numpy()

        pred_mask = (pred_mask - pred_mask.min()) / pred_mask.max()

        mask = np.zeros(torch.sigmoid(curr_pred).shape[0:2], dtype='uint8')
        mask[pred_mask>=threshold] = 255
        masks.append(mask)
        smooth_masks.append(pred_mask)

    return masks, smooth_masks
