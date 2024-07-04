# Generative-Image-Anonymization-Pipeline

## Overview of the pipeline
An initial exploration for using a generative image pipeline for anonymization of datasets for anomaly detection.
1. The pipeline starts by pre-processing the given dataset by running YoloV8 for detecting and tracking pedestrians
2. The tracked bounding boxes are then run through ClipSeg to segment the pedestrians and reconstruct the background
3. The bounding boxes are also given to OpenPose to extract pose skeletonization images
4. The pre-processed images together with CSV containing ids of tracklets together with bounding box sizes and image names are given to the generation pipeline
5. For each track series the skeletonizations are given to a Diffusers animation diffusion model
6. A randomized prompt is generated for each sequence
7. The generated animated frames are segmented and blended into the larger image

## Requirements

- Pytorch > 2.2.0, CUDA 11.8, torchvision
- Diffusers library - [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers) 
- Pandas, numpy
- Pillow, OpenCV
- YoloV8 - [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) 
- ControlNet auxiliary models -  [https://github.com/huggingface/controlnet_aux](https://github.com/huggingface/controlnet_aux) 
- Compel - [https://github.com/damian0815/compel](https://github.com/damian0815/compel)

## Use
- Put the dataset images that you need to anonymize in the Dataset -> training or testing directory
- Run the pre_process_pipeline.py, change the input directories if needed
- The pre-processed images will be put in the pre_processing directory
- Run the generate_pipeline.py, change the input directories if needed
- The generated augmentations will be put in the results directory
