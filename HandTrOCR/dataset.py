import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
import csv
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from transformers import TrOCRProcessor
from pathlib import Path

# Base directory setup
base_dir = Path(__file__).resolve().parent
esposalles_base = base_dir / 'Esposalles'
train_dir = esposalles_base / 'train'
test_dir = esposalles_base / 'test'

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
                degrees=3,  # Reduced for handwritten text
                translate=(0.03, 0.03),  # Reduced translation
                scale=(0.98, 1.02),  # Minimal scaling
                shear=2,  # Reduced shear
                fill=255,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            probability=0.2
        ),
        
        # Perspective distortion (minimal for handwritten text)
        ProbabilisticTransform(
            transforms.RandomPerspective(
                distortion_scale=0.1,  # Reduced distortion
                p=1.0,
                fill=255
            ),
            probability=0.15
        ),
        
        # Blur effects (minimal)
        ProbabilisticTransform(
            transforms.GaussianBlur(
                kernel_size=3,  # Smaller kernel
                sigma=(0.1, 0.5)  # Less blur
            ),
            probability=0.2
        ),
        
        # Brightness and contrast variations
        ProbabilisticTransform(
            BrightnessContrastTransform(
                brightness_range=(0.8, 1.2),  # Reduced range
                contrast_range=(0.8, 1.2)
            ),
            probability=0.3
        ),
        
        # Add minimal noise
        ProbabilisticTransform(
            NoiseTransform(noise_factor=0.03),  # Reduced noise
            probability=0.15
        ),
        
        # Shadow effects (minimal)
        ProbabilisticTransform(
            ShadowTransform(),
            probability=0.1
        ),
        
        # Minimal color jitter
        ProbabilisticTransform(
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.01
            ),
            probability=0.2
        ),
    ])

class SimpleEsposallesDataset(Dataset):
    def __init__(self, data_split='train', augmentation=False):
        """
        Simple OCR-only dataset for Esposalles
        
        Args:
            data_split (str): 'train' or 'test'
            augmentation: Whether to apply image augmentations
        """
        self.data_split = data_split
        self.augmentation = augmentation
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        
        self.image_paths = []
        self.labels = []
        
        # Set up transforms
        if self.augmentation:
            self.transform = get_augmentation_transforms()
        else:
            self.transform = None
        
        if data_split == 'train':
            self._load_train_data()
        elif data_split == 'test':
            self._load_test_data()
        else:
            raise ValueError("data_split must be 'train' or 'test'")
    
    def _load_train_data(self):
        """Load training data from train directory structure"""
        train_path = Path(train_dir)
        
        # Iterate through all record folders (idPage10354_Record1, etc.)
        for record_dir in train_path.iterdir():
            if record_dir.is_dir() and record_dir.name.startswith('idPage'):
                words_dir = record_dir / 'words'
                # Ground truth file naming: idPage10354_Record1_transcription.txt
                ground_truth_file = words_dir / f'{record_dir.name}_transcription.txt'
                
                if words_dir.exists() and ground_truth_file.exists():
                    # Read ground truth file
                    ground_truth_dict = {}
                    with open(ground_truth_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if ':' in line:
                                image_name, text = line.split(':', 1)
                                ground_truth_dict[image_name] = text
                    
                    # Find all image files in words directory
                    for image_file in words_dir.iterdir():
                        if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            image_name = image_file.stem  # filename without extension
                            if image_name in ground_truth_dict:
                                self.image_paths.append(str(image_file))
                                self.labels.append(ground_truth_dict[image_name])
    
    def _load_test_data(self):
        """Load test data from test directory structure with XML ground truth"""
        test_path = Path(test_dir)
        gt_dir = test_path / 'gt'
        
        if not gt_dir.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
        
        # Process each XML file in gt directory
        for xml_file in gt_dir.iterdir():
            if xml_file.suffix.lower() == '.xml':
                try:
                    # Parse XML file
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    # Extract idPage from the XML
                    id_page = None
                    page_element = root.find('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Page')
                    if page_element is not None:
                        for prop in page_element.findall('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Property'):
                            if prop.get('key') == 'idPage':
                                id_page = prop.get('value')
                                break
                    
                    if not id_page:
                        print(f"Warning: Could not find idPage in {xml_file}")
                        continue
                    
                    # Build ground truth dictionary from XML
                    ground_truth_dict = {}
                    
                    # Find all Word elements
                    word_elements = root.findall('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Word')
                    
                    for word_elem in word_elements:
                        word_id = word_elem.get('id')  # e.g., "Record1_Line0_Word0"
                        
                        # Find the Unicode text content
                        unicode_elem = word_elem.find('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Unicode')
                        if unicode_elem is not None and unicode_elem.text:
                            # Convert Word ID to image name format
                            # "Record1_Line0_Word0" -> "idPage10479_Record1_Line0_Word0"
                            image_name = f"idPage{id_page}_{word_id}"
                            ground_truth_dict[image_name] = unicode_elem.text.strip()
                    
                    # Now find corresponding image files for this idPage
                    # Look for folders that start with "idPage{id_page}_"
                    for record_dir in test_path.iterdir():
                        if (record_dir.is_dir() and 
                            record_dir.name.startswith(f'idPage{id_page}_')):
                            
                            words_dir = record_dir / 'words'
                            if words_dir.exists():
                                # Find all image files
                                for image_file in words_dir.iterdir():
                                    if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                                        image_name = image_file.stem  # filename without extension
                                        if image_name in ground_truth_dict:
                                            self.image_paths.append(str(image_file))
                                            self.labels.append(ground_truth_dict[image_name])
                
                except ET.ParseError as e:
                    print(f"Error parsing XML file {xml_file}: {e}")
                except Exception as e:
                    print(f"Error processing XML file {xml_file}: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert('RGB')
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
            'url': image_path
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

def load_simple_esposalles_datasets(augmentation=False):
    """
    Load train and test datasets for simple OCR
    
    Args:
        augmentation: Whether to apply augmentation to training data
    
    Returns:
        train_dataset, test_dataset
    """
    train_dataset = SimpleEsposallesDataset('train', augmentation=augmentation)
    test_dataset = SimpleEsposallesDataset('test', augmentation=False)  # Never augment test data
    
    return train_dataset, test_dataset