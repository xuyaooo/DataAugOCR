import torch
import random
import numpy as np
import csv
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance
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
esposalles_base = base_dir / 'Esposalles'
train_dir = esposalles_base / 'train'
test_dir = esposalles_base / 'test'


class STRAugTransform:
    """STRAug group-based transform for handwritten text"""
    
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


class SimpleEsposallesDataset(Dataset):
    def __init__(self, data_split='train', augmentation=False, straug_N=3):
        """
        Simple OCR dataset for Esposalles with STRAug support
        
        Args:
            data_split (str): 'train' or 'test'
            augmentation: Whether to apply STRAug augmentations
            straug_N: Number of STRAug groups to select (1-8)
        """
        self.data_split = data_split
        self.augmentation = augmentation
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        
        self.image_paths = []
        self.labels = []
        
        # Set up STRAug transforms
        if self.augmentation:
            self.transform = STRAugTransform(N=straug_N)
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
        
        # Apply STRAug if enabled
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


def load_simple_esposalles_datasets(augmentation=False, straug_N=3):
    """
    Load train and test datasets for simple OCR with STRAug
    
    Args:
        augmentation: Whether to apply STRAug to training data
        straug_N: Number of STRAug groups to select (1-8)
    
    Returns:
        train_dataset, test_dataset
    """
    train_dataset = SimpleEsposallesDataset('train', augmentation=augmentation, straug_N=straug_N)
    test_dataset = SimpleEsposallesDataset('test', augmentation=False)  # Never augment test data
    
    return train_dataset, test_dataset


# Keep the same interface as your original code
def load_datasets(augmentation=False, straug_N=3):
    """
    Load datasets with the same interface as your original code
    
    Args:
        augmentation: Whether to apply augmentation to training data
        straug_N: Number of STRAug groups to select (1-8)
    
    Returns:
        train_dataset, test_dataset
    """
    return load_simple_esposalles_datasets(augmentation=augmentation, straug_N=straug_N)