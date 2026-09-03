import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class ImageRestorationDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, gt_crop_size=128, synthesize_prob=0.5, is_val=False):
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.gt_crop_size = gt_crop_size
        self.synthesize_prob = synthesize_prob
        self.is_val = is_val
        
        # We assume 1-to-1 matching filenames
        self.filenames = sorted(os.listdir(self.gt_dir))
        
        # Calculate scale factor dynamically from the first image pair
        if len(self.filenames) > 0:
            sample_gt = np.load(os.path.join(self.gt_dir, self.filenames[0]))
            sample_noisy = np.load(os.path.join(self.noisy_dir, self.filenames[0]))
            
            # Assuming square images for scale factor calculation
            if sample_gt.shape[0] % sample_noisy.shape[0] != 0:
                raise ValueError(f"GT shape {sample_gt.shape} is not perfectly divisible by NoisyLR shape {sample_noisy.shape}")
            self.scale = sample_gt.shape[0] // sample_noisy.shape[0]
            
            if self.gt_crop_size % self.scale != 0:
                raise ValueError(f"gt_crop_size {self.gt_crop_size} is not perfectly divisible by scale {self.scale}")
            self.noisy_crop_size = self.gt_crop_size // self.scale
        else:
            self.scale = 2 # default fallback
            self.noisy_crop_size = 64

    def synthesize_noisy(self, gt_tensor):
        """
        Synthesize noisy image from GT image by applying:
        - Speckle noise (multiplicative)
        - Gaussian noise (additive)
        - Downsampling
        in randomized order to avoid overfitting.
        """
        noisy = gt_tensor.clone()
        
        # Heuristic noise parameters for robust synthesis
        speckle_std = random.uniform(0.1, 0.4)
        gaussian_std = random.uniform(0.05, 0.2)
        
        def apply_speckle(img):
            noise = torch.randn_like(img) * speckle_std
            return img + img * noise
            
        def apply_gaussian(img):
            noise = torch.randn_like(img) * gaussian_std
            return img + noise
            
        def apply_downsample(img):
            # Using bilinear interpolation for downsampling, expects (N, C, H, W)
            return torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=1.0/self.scale, mode='bilinear', align_corners=False).squeeze(0)
            
        # Randomize order of operations
        operations = [apply_speckle, apply_gaussian, apply_downsample]
        random.shuffle(operations)
        
        for op in operations:
            noisy = op(noisy)
            
        return noisy

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        gt_path = os.path.join(self.gt_dir, filename)
        gt_arr = np.load(gt_path).astype(np.float32)[np.newaxis, ...]
        gt_tensor = torch.from_numpy(gt_arr)
        
        is_synthetic = False
        if random.random() < self.synthesize_prob:
            # Generate synthetic NoisyLR dynamically
            noisy_tensor = self.synthesize_noisy(gt_tensor)
            is_synthetic = True
        else:
            # Load real NoisyLR from disk
            noisy_path = os.path.join(self.noisy_dir, filename)
            noisy_arr = np.load(noisy_path).astype(np.float32)[np.newaxis, ...]
            noisy_tensor = torch.from_numpy(noisy_arr)
        
        # Random Crop (only if not validation)
        if not self.is_val:
            h_noisy, w_noisy = noisy_tensor.shape[1], noisy_tensor.shape[2]
            
            if h_noisy >= self.noisy_crop_size and w_noisy >= self.noisy_crop_size:
                top_noisy = random.randint(0, h_noisy - self.noisy_crop_size)
                left_noisy = random.randint(0, w_noisy - self.noisy_crop_size)
                
                top_gt = top_noisy * self.scale
                left_gt = left_noisy * self.scale
                
                noisy_tensor = noisy_tensor[:, top_noisy:top_noisy+self.noisy_crop_size, left_noisy:left_noisy+self.noisy_crop_size]
                gt_tensor = gt_tensor[:, top_gt:top_gt+self.gt_crop_size, left_gt:left_gt+self.gt_crop_size]
                
            # Random augmentations (geometric only to preserve noise statistics)
            # Random horizontal flip
            if random.random() > 0.5:
                noisy_tensor = TF.hflip(noisy_tensor)
                gt_tensor = TF.hflip(gt_tensor)
                
            # Random vertical flip
            if random.random() > 0.5:
                noisy_tensor = TF.vflip(noisy_tensor)
                gt_tensor = TF.vflip(gt_tensor)
                
            # Random 90-degree rotations
            rotations = random.choice([0, 1, 2, 3])
            if rotations > 0:
                noisy_tensor = torch.rot90(noisy_tensor, k=rotations, dims=[1, 2])
                gt_tensor = torch.rot90(gt_tensor, k=rotations, dims=[1, 2])
            
        return noisy_tensor, gt_tensor, filename, is_synthetic
