import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms




class PeakTracesDataset(Dataset):
    def __init__(self, data_dir, humans_only=True, augment=False):
        
        self.img_dir = data_dir
        annotations_file = os.path.join(self.img_dir,"target_labels.csv")
        
        
        print(self.img_dir)
        self.image_data = pd.read_csv(annotations_file) # Video  Track  Class  NumFrames  NumFeatures

        self.image_files = []

        for row in self.image_data.iterrows():
            filename = "{0}_{1}.png".format(str(row[1].Video).zfill(2), str(row[1].Track).zfill(6))

            
            assert os.path.exists(os.path.join(self.img_dir,filename)), "File '{0}' does not exist.".format(filename)
            self.image_files.append(filename)

        if humans_only == True:
            class_arr = np.array(self.image_data.get('Class').values, dtype=np.int32)
            target_args = np.where(class_arr == 0)[0]

            self.image_files = list(np.array(self.image_files)[target_args])
            self.image_data = self.image_data.iloc[target_args]

        self.image_labels = torch.nn.functional.one_hot(torch.tensor(self.image_data.get('Class').values)) * 1.0
        self.augment = augment


    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        
        assert idx < len(self.image_files), "len(image_files) = {0}, idx = {1}".format(len(self.image_files), idx)
 
        # image = torch.tensor(np.array(read_image(os.path.join(self.img_dir,self.image_files[idx]), mode=ImageReadMode.UNCHANGED), dtype=np.uint8))
        
        image = torch.tensor(np.array(read_image(os.path.join(self.img_dir,self.image_files[idx]), mode=ImageReadMode.GRAY), dtype=np.float32) / 255.0)
        image_data = self.image_data.iloc[idx]
        label = self.image_labels[idx]

        curr_filename = self.image_files[idx]
           
        num_frames, num_features = image_data.get('NumFrames').item(), image_data.get('NumFeatures').item()
        assert (num_features == image.shape[2]) and (num_frames == image.shape[1]), "(num_features ({0}) != image.shape[2] ({1})) or (num_frames  ({2}) != image.shape[1] ({3}))".format(num_features, image.shape[2], num_frames, image.shape[1])
       
        num_features = int(0.5 * num_features)
        transform = transforms.Compose([ transforms.Resize((num_features,num_features), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True) ])
        image = transform(image)
 
        if self.augment == True:
            image = transforms.functional.vflip(image)
 
       
       
        return image, label, curr_filename
    



def get_data(data_dir, batch_size, humans_only=True, augment=True, shuffle=False):
    
    if augment == True:
        dataset_init = PeakTracesDataset(data_dir, humans_only=humans_only)
        dataset_augm = PeakTracesDataset(data_dir, humans_only=humans_only, augment=augment)
        dataset = torch.utils.data.ConcatDataset([ dataset_init, dataset_augm ])

    else:
        dataset = PeakTracesDataset(data_dir, humans_only=humans_only)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader


            
