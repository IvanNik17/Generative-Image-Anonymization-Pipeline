from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
import torch
from PIL import Image
import cv2
import numpy as np


def initializeDiffusers():

    open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

    return open_pose

def calculateOpenPose(image, openpose_model, resolution = 128, dresolution = 512):
    image = Image.fromarray(np.uint8(image)) 
    image, poseCurr = openpose_model(image, detect_resolution=dresolution, image_resolution=resolution)

    return image, poseCurr



def rescaleImage(image, scale, bigger = True):
    if bigger:
        resized_image = cv2.resize(np.array(image),(0,0), fx = scale, fy = scale)
    else:
        resized_image = cv2.resize(np.array(image),(0,0), fx = 1/scale, fy = 1/scale)

    return resized_image

def resizeImage(image, res):
    image_resize = cv2.resize(image, res)

    return image_resize