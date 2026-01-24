import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from qa_new import generate_qas, CATEGORY_GENERATORS, BIG_CATEGORIES
from transformers import TrOCRProcessor
from pathlib import Path

# Base directory setup
base_dir = Path(__file__).resolve().parent
img_base = base_dir / 'WordArt'
train_set = str(img_base / 'train_label.txt')
test_set = str(img_base / 'test_label.txt')
img_base = str(img_base) + "/"

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


class SimpleArtDataset(Dataset):
    def __init__(self, file_path, categories=None, category_percentages=None):
        """
        Simplified dataset for the modified TrOCR model that takes text inputs directly
        
        Args:
            file_path (str): Path to the label file (train_label.txt or test_label.txt)
            categories (list): List of QA categories to generate (original mode)
            category_percentages (dict): Percentages for probabilistic mode (new mode)
                If both are provided, category_percentages takes precedence
        """
        self.urls = []
        self.labels = []
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
        
        # Determine mode: probabilistic or fixed categories
        if category_percentages is not None:
            self.mode = 'probabilistic'
            self.question_generator = ProbabilisticQuestionGenerator(category_percentages)
            self.categories = None
        else:
            self.mode = 'fixed'
            self.categories = categories if categories is not None else [0]
            self.question_generator = None
        
        # Load data from file
        with open(file_path, 'r', encoding='utf-8') as _f:
            for line in _f:
                parts = line.strip().split(' ')
                if len(parts) >= 2:
                    url = parts[0]
                    text = ' '.join(parts[1:])
                    self.urls.append(url)
                    self.labels.append(text)

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        img = Image.open(f'{img_base}{url}').convert('RGB')
        text = self.labels[idx]

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
            urls.append(url)  # One URL per QA pair

        return {
            'pixel_values': pixel_values_list,
            'questions': questions,
            'answers': answers,
            'urls': urls  # Now returns list of URLs (one per QA pair)
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
        all_urls.extend(item['urls'])  # Just extend the list directly

    # Stack all pixel_values into a single tensor
    stacked_pixel_values = torch.stack(all_pixel_values)

    return {
        'pixel_values': stacked_pixel_values,
        'questions': all_questions,
        'answers': all_answers,
        'urls': all_urls
    }


def load_simple_data(train_categories=None):
    """
    Load datasets with simplified structure (ORIGINAL FUNCTION - unchanged)
    """
    if train_categories is None:
        train_categories = list(CATEGORY_GENERATORS.keys())

    data_train = SimpleArtDataset(train_set, categories=train_categories)
    data_test = SimpleArtDataset(test_set, categories=train_categories)

    # Recognition-only datasets (category 0 only)
    data_train_rec = SimpleArtDataset(train_set, categories=[0])
    data_test_rec = SimpleArtDataset(test_set, categories=[0])

    return data_train, data_test, data_train_rec, data_test_rec


def load_simple_data_probabilistic(category_percentages=None):
    """
    Load WordArt datasets with probabilistic question generation (NEW FUNCTION)
    
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
    data_train_prob = SimpleArtDataset(train_set, category_percentages=category_percentages)
    data_test_prob = SimpleArtDataset(test_set, category_percentages=category_percentages)
    
    # Recognition-only datasets (for evaluation)
    data_train_rec = SimpleArtDataset(train_set, categories=[0])
    data_test_rec = SimpleArtDataset(test_set, categories=[0])
    
    return data_train_prob, data_test_prob, data_train_rec, data_test_rec