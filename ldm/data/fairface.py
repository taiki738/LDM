import os
import numpy as np
import pandas as pd
import PIL
from PIL import Image # Removed Resampling from here
from torch.utils.data import Dataset
from torchvision import transforms

class FairFaceBase(Dataset):
    def __init__(self,
                 data_root,
                 csv_file,
                 size=128, # FairFace images will be resized to 128x128 as per project summary
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self.csv_file = csv_file
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Filter for rows where 'file' column starts with 'train/' or 'val/'
        # Assuming data_root points to the directory containing 'train' and 'val' subfolders
        # The 'file' column in CSV is like 'train/image.jpg' or 'val/image.jpg'
        self.image_paths = [os.path.join(self.data_root, row['file']) for index, row in df.iterrows()]
        
        # Optionally, store labels if needed for conditional generation later
        # self.genders = [row['gender'] for index, row in df.iterrows()]
        # self.races = [row['race'] for index, row in df.iterrows()]

        self._length = len(self.image_paths)

        self.size = size
        self.interpolation = {"linear": Image.Resampling.BILINEAR, # Changed LINEAR to BILINEAR
                              "bilinear": Image.Resampling.BILINEAR,
                              "bicubic": Image.Resampling.BICUBIC,
                              "lanczos": Image.Resampling.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        # This part ensures images are square before resizing
        img = np.array(image).astype(np.uint8)
        
        # Calculate aspect ratio
        h, w = img.shape[0], img.shape[1]
        
        # Resize/crop to the target size while maintaining aspect ratio
        if h != self.size or w != self.size:
            # For FairFace, images are usually 200x200 or similar, but
            # if they are not square, we should center crop or pad.
            # Assuming square input is preferred, let's take a center crop
            # consistent with LDM's internal practices for image datasets.
            # For now, let's just resize directly. If aspect ratio issues
            # arise, we can add center cropping.
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        
        # Normalize to [-1, 1]
        example = {"image": (image / 127.5 - 1.0).astype(np.float32)}
        
        # If conditional generation is needed later, add labels here
        # example["class_label"] = self.genders[i] # or a mapped ID
        return example


class FairFaceTrain(FairFaceBase):
    def __init__(self, **kwargs):
        # The data_root is expected to be 'latent-diffusion/data/fairface' (symlink to fairface_original)
        # The csv_file should point to the correct CSV relative to the project root
        super().__init__(csv_file="data/fairface_label_train.csv",
                         data_root="data/fairface", # This is the symlinked directory
                         **kwargs)


class FairFaceValidation(FairFaceBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(csv_file="data/fairface_label_val.csv",
                         data_root="data/fairface", # This is the symlinked directory
                         flip_p=flip_p, # No flip for validation
                         **kwargs)