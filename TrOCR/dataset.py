import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from transformers import TrOCRProcessor
from pathlib import Path

# Base directory setup
base_dir = Path(__file__).resolve().parent
img_base = base_dir / 'WordArt'

train_set = str(img_base / 'train_label.txt')
test_set = str(img_base / 'test_label.txt')
img_base = str(img_base) + "/"

class ProbabilisticTransform:
    """Wrapper to apply transforms with specific probabilities"""
    def __init__(self, transform, probability):
        self.transform = transform
        self.probability = probability
    
    def __call__(self, img):
        if torch.rand(1).item() < self.probability:
            return self.transform(img)
        return img

class NoiseTransform:
    """Add random noise to simulate scanning artifacts"""
    def __init__(self, noise_factor=0.05):
        self.noise_factor = noise_factor
    
    def __call__(self, img):
        img_array = np.array(img)
        noise = np.random.normal(0, self.noise_factor * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

class BrightnessContrastTransform:
    """Adjust brightness and contrast to simulate different lighting conditions"""
    def __init__(self, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, img):
        # Random brightness
        brightness_factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        
        # Random contrast
        contrast_factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        
        return img

class ShadowTransform:
    """Add shadow effects to simulate uneven lighting"""
    def __call__(self, img):
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Create a gradient shadow
        shadow_intensity = random.uniform(0.3, 0.7)
        shadow_direction = random.choice(['top', 'bottom', 'left', 'right'])
        
        if shadow_direction == 'top':
            gradient = np.linspace(shadow_intensity, 1.0, height)
            shadow_mask = np.repeat(gradient[:, np.newaxis], width, axis=1)
        elif shadow_direction == 'bottom':
            gradient = np.linspace(1.0, shadow_intensity, height)
            shadow_mask = np.repeat(gradient[:, np.newaxis], width, axis=1)
        elif shadow_direction == 'left':
            gradient = np.linspace(shadow_intensity, 1.0, width)
            shadow_mask = np.repeat(gradient[np.newaxis, :], height, axis=0)
        else:  # right
            gradient = np.linspace(1.0, shadow_intensity, width)
            shadow_mask = np.repeat(gradient[np.newaxis, :], height, axis=0)
        
        if len(img_array.shape) == 3:
            shadow_mask = np.stack([shadow_mask] * 3, axis=-1)
        
        shadowed_img = (img_array * shadow_mask).astype(np.uint8)
        return Image.fromarray(shadowed_img)

def get_augmentation_transforms():
    """Enhanced OCR augmentation transforms"""
    return transforms.Compose([
        # Geometric transformations
        ProbabilisticTransform(
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=3,
                fill=255,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            probability=0.3
        ),
        
        # Perspective distortion
        ProbabilisticTransform(
            transforms.RandomPerspective(
                distortion_scale=0.15,
                p=1.0,
                fill=255
            ),
            probability=0.25
        ),
        
        # Blur effects
        ProbabilisticTransform(
            transforms.GaussianBlur(
                kernel_size=5,
                sigma=(0.1, 1.0)
            ),
            probability=0.3
        ),
        
        # Elastic transform for paper warping
        ProbabilisticTransform(
            transforms.ElasticTransform(
                alpha=8.0,
                sigma=1.5,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=255
            ),
            probability=0.2
        ),
        
        # Brightness and contrast variations
        ProbabilisticTransform(
            BrightnessContrastTransform(
                brightness_range=(0.6, 1.4),
                contrast_range=(0.6, 1.4)
            ),
            probability=0.4
        ),
        
        # Add noise
        ProbabilisticTransform(
            NoiseTransform(noise_factor=0.08),
            probability=0.25
        ),
        
        # Shadow effects
        ProbabilisticTransform(
            ShadowTransform(),
            probability=0.2
        ),
        
        # Color jitter
        ProbabilisticTransform(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.02
            ),
            probability=0.3
        ),
    ])

class ArtDataset(Dataset):
    def __init__(self, file_path, augmentation=False):
        """
        Simple dataset for OCR with optional augmentation
        
        Args:
            file_path: Path to label file (train_label.txt or test_label.txt)
            augmentation: Whether to apply image augmentations
        """
        self.augmentation = augmentation
        self.urls = []
        self.labels = []
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
        
        # Set up transforms
        if self.augmentation:
            self.transform = get_augmentation_transforms()
        else:
            self.transform = None
        
        # Load data from label file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 2:
                    url = parts[0]
                    text = ' '.join(parts[1:])
                    self.urls.append(url)
                    self.labels.append(text)

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        # Load image
        url = self.urls[idx]
        img = Image.open(f'{img_base}{url}').convert('RGB')
        text = self.labels[idx]
        
        # Apply augmentation if enabled
        if self.augmentation and self.transform:
            img = self.transform(img)
        
        # Process image with TrOCR processor
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)  # Remove batch dimension
        
        return {
            'image': pixel_values,
            'text': text,
            'url': url
        }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    urls = [item['url'] for item in batch]
    
    return {
        'images': images,
        'texts': texts,
        'urls': urls
    }

def load_datasets(augmentation=False):
    """
    Load train and test datasets
    
    Args:
        augmentation: Whether to apply augmentation to training data
    
    Returns:
        train_dataset, test_dataset
    """
    train_dataset = ArtDataset(train_set, augmentation=augmentation)
    test_dataset = ArtDataset(test_set, augmentation=False)  # Never augment test data
    
    return train_dataset, test_dataset
