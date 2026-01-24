import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor
from pathlib import Path

# STRAug imports
from straug.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from straug.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from straug.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color
from straug.warp import Curve, Distort, Stretch
from straug.weather import Fog, Snow, Frost, Rain, Shadow

# Base directory setup
base_dir = Path(__file__).resolve().parent
img_base = base_dir / 'WordArt'

train_set = str(img_base / 'train_label.txt')
test_set = str(img_base / 'test_label.txt')
img_base = str(img_base) + "/"


class STRAugTransform:
    """STRAug group-based transform with fixed prob=0.5 and random magnitude 0-2"""
    
    def __init__(self, N=3):
        """
        Args:
            N: Number of groups to select (1-8)
        """
        self.N = N
        
        # 8 groups as specified in STRAug paper
        self.groups = {
            'Warp': [Curve(), Distort(), Stretch()],
            'Geometry': [Rotate(), Perspective(), Shrink(), TranslateX(), TranslateY()],
            'Pattern': [VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()],
            'Noise': [GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()],
            'Blur': [GaussianBlur(), DefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()],
            'Weather': [Fog(), Snow(), Frost(), Rain(), Shadow()],
            'Camera': [Contrast(), Brightness(), JpegCompression(), Pixelate()],
            'Process': [Posterize(), Solarize(), Invert(), Equalize(), AutoContrast(), Sharpness(), Color()]
        }
        
        total_ops = sum(len(ops) for ops in self.groups.values())
        print(f"STRAug: {len(self.groups)} groups, {total_ops} operations, N={N}")
    
    def __call__(self, img):
        """Apply N random groups with 50% prob per function and random magnitude 0-2"""
        # Select N random groups
        group_names = list(self.groups.keys())
        selected_groups = random.sample(group_names, min(self.N, len(group_names)))
        
        # Apply functions from selected groups
        for group_name in selected_groups:
            for op in self.groups[group_name]:
                # 50% probability for each function
                if random.random() < 0.5:
                    try:
                        # Random magnitude 0-2
                        mag = random.randint(0, 2)
                        img = op(img, mag=mag)
                    except:
                        continue
        
        return img


class ArtDataset(Dataset):
    def __init__(self, file_path, augmentation=False, straug_N=3):
        """
        Args:
            file_path: Path to label file
            augmentation: Whether to use STRAug
            straug_N: Number of groups to select (1-8)
        """
        self.augmentation = augmentation
        self.urls = []
        self.labels = []
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
        
        # Setup STRAug
        if augmentation:
            self.transform = STRAugTransform(N=straug_N)
        else:
            self.transform = None
        
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 2:
                    url = parts[0]
                    text = ' '.join(parts[1:])
                    self.urls.append(url)
                    self.labels.append(text)
        
        print(f"Loaded {len(self.urls)} samples, augmentation={augmentation}")

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        # Load image
        url = self.urls[idx]
        img = Image.open(f'{img_base}{url}').convert('RGB')
        text = self.labels[idx]
        
        # Apply STRAug
        if self.augmentation and self.transform:
            img = self.transform(img)
        
        # Process with TrOCR
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)
        
        return {
            'image': pixel_values,
            'text': text,
            'url': url
        }


def collate_fn(batch):
    """Custom collate function"""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    urls = [item['url'] for item in batch]
    
    return {
        'images': images,
        'texts': texts,
        'urls': urls
    }


def load_datasets(augmentation=False, straug_N=3):
    """
    Load datasets
    
    Args:
        augmentation: Whether to use STRAug on training data
        straug_N: Number of groups to select (1-8)
    """
    train_dataset = ArtDataset(train_set, augmentation=augmentation, straug_N=straug_N)
    test_dataset = ArtDataset(test_set, augmentation=False)
    
    return train_dataset, test_dataset