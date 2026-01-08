import os
import numpy as np
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FFHQSurveyBase(Dataset):
    """
    Data loader for FFHQ Survey Dataset (Male only for now).
    Directory structure:
        data_root/
            male/
                OK_4.0/
                not-OK_2.0/
            female/
                ...
    
    This loader ignores the distinction between OK and not-OK, 
    treating all images in the selected sex folder as a single dataset.
    """
    def __init__(self,
                 size=256, # Default to 256 to match pre-trained model
                 interpolation="bicubic",
                 flip_p=0.5,
                 split="train",
                 ratio=0.9,
                 seed=42,
                 sex="male" 
                 ):
        self.size = size
        self.interpolation = {"linear": Image.Resampling.BILINEAR,
                              "bilinear": Image.Resampling.BILINEAR,
                              "bicubic": Image.Resampling.BICUBIC,
                              "lanczos": Image.Resampling.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        
        self.data_root = Path("data/ffhq_survey")
        self.image_paths = []
        
        if not self.data_root.exists():
            # Fallback to absolute path
            self.data_root = Path("/home/tm/img-science/github/latent-diffusion/data/ffhq_survey")

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        target_sexes = [sex] if sex != "all" else ["male", "female"]
        target_labels = ["OK_4.0", "not-OK_2.0"] # Load both groups

        for s in target_sexes:
            for label_str in target_labels:
                dir_path = self.data_root / s / label_str
                if dir_path.exists():
                    files = sorted(list(dir_path.glob("*.png")))
                    for f in files:
                        self.image_paths.append(str(f))

        # Shuffle and Split
        random.seed(seed)
        random.shuffle(self.image_paths)
        
        split_idx = int(len(self.image_paths) * ratio)
        
        if split == "train":
            self.data = self.image_paths[:split_idx]
        elif split == "val":
            self.data = self.image_paths[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}")
            
        self._length = len(self.data)
        print(f"FFHQSurveyDataset ({split}, sex={sex}): Loaded {self._length} images (Unconditional).")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image_path = self.data[i]
        
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        
        # Normalize to [-1, 1]
        # Return simple dict for unconditional
        return {
            "image": (image / 127.5 - 1.0).astype(np.float32)
        }

class FFHQSurveyTrain(FFHQSurveyBase):
    def __init__(self, **kwargs):
        super().__init__(split="train", **kwargs)

class FFHQSurveyValidation(FFHQSurveyBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(split="val", flip_p=flip_p, **kwargs)
