import torch
import csv
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from qa_new import generate_qas, CATEGORY_GENERATORS, BIG_CATEGORIES
from transformers import TrOCRProcessor
from pathlib import Path
import os
import re
import random

# Base directory setup
base_dir = Path(__file__).resolve().parent
esposalles_base = base_dir / 'Esposalles'
train_dir = esposalles_base / 'train'
test_dir = esposalles_base / 'test'

class ProbabilisticQuestionGenerator:
    """
    A class to handle probabilistic question generation where recognition is always included
    and exactly one of the 4 big categories is selected based on percentages that sum to 100%
    """
    
    def __init__(self, category_percentages=None):
        """
        Initialize the probabilistic question generator
        
        Args:
            category_percentages (dict): Dictionary mapping category names to percentages (must sum to 100)
                Example: {
                    "CHARACTER_PRESENCE": 25,    # 25% chance for categories 1,2
                    "POSITIONAL_ANALYSIS": 35,   # 35% chance for categories 3,4
                    "STRUCTURAL_ANALYSIS": 20,   # 20% chance for categories 5,6
                    "BOUNDARY_ANALYSIS": 20      # 20% chance for categories 7,8
                }
                Note: Recognition (category 0) is ALWAYS included
        """
        if category_percentages is None:
            # Default percentages - equal distribution
            category_percentages = {
                "CHARACTER_PRESENCE": 25,
                "POSITIONAL_ANALYSIS": 25,
                "STRUCTURAL_ANALYSIS": 25,
                "BOUNDARY_ANALYSIS": 25
            }
        
        # Validate that percentages sum to 100
        total_percentage = sum(category_percentages.values())
        if abs(total_percentage - 100) > 0.01:  # Allow for small floating point errors
            raise ValueError(f"Category percentages must sum to 100, but got {total_percentage}")
        
        self.category_percentages = category_percentages
        
        # Mapping from big category names to category IDs
        self.big_category_mapping = {
            "CHARACTER_PRESENCE": BIG_CATEGORIES["CHARACTER_PRESENCE"],  # [1, 2]
            "POSITIONAL_ANALYSIS": BIG_CATEGORIES["POSITIONAL_ANALYSIS"],  # [3, 4]
            "STRUCTURAL_ANALYSIS": BIG_CATEGORIES["STRUCTURAL_ANALYSIS"],  # [5, 6]
            "BOUNDARY_ANALYSIS": BIG_CATEGORIES["BOUNDARY_ANALYSIS"]   # [7, 8]
        }
        
        # Create cumulative distribution for sampling
        self.categories = list(self.category_percentages.keys())
        self.weights = list(self.category_percentages.values())
    
    def sample_categories_for_word(self, word):
        """
        Sample which categories to use for a given word
        Always includes recognition (0) + exactly one of the 4 big categories
        
        Args:
            word (str): The input word
            
        Returns:
            list: List of category IDs to use for this word
        """
        # Always start with recognition
        selected_categories = [0]
        
        # Sample exactly one big category based on the percentages
        selected_big_category = random.choices(self.categories, weights=self.weights, k=1)[0]
        
        # Add the category IDs from the selected big category
        category_ids = self.big_category_mapping[selected_big_category]
        selected_categories.extend(category_ids)
        
        # Sort for consistency
        selected_categories = sorted(selected_categories)
        
        return selected_categories
    
    def generate_probabilistic_qas(self, word):
        """
        Generate QA pairs for a word using probabilistic category selection
        
        Args:
            word (str): The input word
            
        Returns:
            list: List of (question, answer) tuples
        """
        selected_categories = self.sample_categories_for_word(word)
        
        questions_answers = []
        for cat_id in selected_categories:
            if cat_id in CATEGORY_GENERATORS:
                try:
                    qa_pair = CATEGORY_GENERATORS[cat_id](word)
                    questions_answers.append(qa_pair)
                except Exception as e:
                    print(f"Error generating QA for category {cat_id} and word '{word}': {e}")
                    # Fallback to recognition if there's an error
                    if cat_id != 0:
                        qa_pair = CATEGORY_GENERATORS[0](word)
                        questions_answers.append(qa_pair)
        
        return questions_answers


class EsposallesDataset(Dataset):
    def __init__(self, data_split='train', categories=None, category_percentages=None):
        """
        Dataset for the Esposalles dataset with the new directory structure
        
        Args:
            data_split (str): 'train' or 'test'
            categories (list): List of QA categories to generate (original mode)
            category_percentages (dict): Percentages for probabilistic mode (new mode)
                If both are provided, category_percentages takes precedence
        """
        self.data_split = data_split
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        
        # Determine mode: probabilistic or fixed categories
        if category_percentages is not None:
            self.mode = 'probabilistic'
            self.question_generator = ProbabilisticQuestionGenerator(category_percentages)
            self.categories = None
        else:
            self.mode = 'fixed'
            self.categories = categories if categories is not None else [0]
            self.question_generator = None
        
        self.image_paths = []
        self.labels = []
        
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
        image_path = self.image_paths[idx]
        text = self.labels[idx]
        
        # Load and process image
        img = Image.open(image_path).convert('RGB')
        
        # Generate QA pairs based on mode
        if self.mode == 'probabilistic':
            qas = self.question_generator.generate_probabilistic_qas(text)
        else:
            qas = generate_qas(text, self.categories)
        
        # Prepare data for each QA pair
        pixel_values_list = []
        questions = []
        answers = []
        urls = []
        
        for question, answer in qas:
            # Process image to get pixel_values using TrOCR processor
            inputs = self.processor(images=img, return_tensors="pt")
            pixel_values = inputs.pixel_values.squeeze(0)  # Remove batch dimension
            
            pixel_values_list.append(pixel_values)
            questions.append(question)
            answers.append(answer)
            urls.append(image_path)  # Use full path as URL
        
        return {
            'pixel_values': pixel_values_list,
            'questions': questions,
            'answers': answers,
            'urls': urls
        }

def simple_collate_fn(batch):
    """
    Simple collate function that flattens the batch and stacks pixel_values
    """
    all_pixel_values = []
    all_questions = []
    all_answers = []
    all_urls = []
    
    for item in batch:
        all_pixel_values.extend(item['pixel_values'])
        all_questions.extend(item['questions'])
        all_answers.extend(item['answers'])
        all_urls.extend(item['urls'])
    
    # Stack all pixel_values into a single tensor
    stacked_pixel_values = torch.stack(all_pixel_values)
    
    return {
        'pixel_values': stacked_pixel_values,
        'questions': all_questions,
        'answers': all_answers,
        'urls': all_urls
    }

def load_esposalles_data(train_categories=None):
    """
    Load Esposalles datasets with the new directory structure (original function)
    
    Args:
        train_categories (list): List of categories for QA generation
    
    Returns:
        tuple: (data_train, data_test, data_train_rec, data_test_rec)
    """
    if train_categories is None:
        train_categories = list(CATEGORY_GENERATORS.keys())
    
    # Full datasets with all categories
    data_train = EsposallesDataset('train', categories=train_categories)
    data_test = EsposallesDataset('test', categories=train_categories)
    
    # Recognition-only datasets (category 0 only)
    data_train_rec = EsposallesDataset('train', categories=[0])
    data_test_rec = EsposallesDataset('test', categories=[0])
    
    return data_train, data_test, data_train_rec, data_test_rec

def load_esposalles_data_probabilistic(category_percentages=None):
    """
    Load Esposalles datasets with probabilistic question generation (NEW FUNCTION)
    
    Args:
        category_percentages (dict): Percentages for each big category (must sum to 100)
            Example: {
                "CHARACTER_PRESENCE": 25,
                "POSITIONAL_ANALYSIS": 35, 
                "STRUCTURAL_ANALYSIS": 20,
                "BOUNDARY_ANALYSIS": 20
            }
    
    Returns:
        tuple: (data_train_prob, data_test_prob, data_train_rec, data_test_rec)
    """
    if category_percentages is None:
        # Default equal distribution
        category_percentages = {
            "CHARACTER_PRESENCE": 25,
            "POSITIONAL_ANALYSIS": 25,
            "STRUCTURAL_ANALYSIS": 25,
            "BOUNDARY_ANALYSIS": 25
        }
    
    # Probabilistic datasets
    data_train_prob = EsposallesDataset('train', category_percentages=category_percentages)
    data_test_prob = EsposallesDataset('test', category_percentages=category_percentages)
    
    # Recognition-only datasets (for evaluation)
    data_train_rec = EsposallesDataset('train', categories=[0])
    data_test_rec = EsposallesDataset('test', categories=[0])
    
    return data_train_prob, data_test_prob, data_train_rec, data_test_rec

# Legacy function for backward compatibility
def load_simple_data(train_categories=None):
    """
    Legacy function name for backward compatibility
    """
    return load_esposalles_data(train_categories)
