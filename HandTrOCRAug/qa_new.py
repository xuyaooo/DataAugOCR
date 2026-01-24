import random

digits = [str(i) for i in range(10)]
lowercase_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
special_char = ['#']
chars = digits + lowercase_letters + special_char

# =============================================================================
# STANDALONE CATEGORY: BASIC RECOGNITION
# =============================================================================

def generate_category_0_recognition(word):
    """Category 0: Basic Recognition/Identification (STANDALONE)"""
    return ("what is this word?", word)

# =============================================================================
# BIG CATEGORY 1: CHARACTER PRESENCE
# =============================================================================

def generate_category_1_existence(word):
    """Category 1: Character existence (yes/no)"""
    unique_chars = list(set(word))
    positive_char = random.choice(unique_chars)
    
    possible_negative_chars = list(set(chars) - set(unique_chars))
    if not possible_negative_chars:
        return (f"is the character '{positive_char}' in this word?", 'yes')
    negative_char = random.choice(possible_negative_chars)
    
    questions = [
        (f"is the character '{positive_char}' in this word?", 'yes'),
        (f"is the character '{negative_char}' in this word?", 'no')
    ]
    return random.choice(questions)

def generate_category_2_quantity(word):
    """Category 2: Character frequency/count"""
    char_counts = {char: word.count(char) for char in set(word)}
    positive_char = random.choice(list(char_counts.keys()))

    possible_negative_chars = list(set(chars) - set(word))
    if not possible_negative_chars:
        return (f"how many times does the character '{positive_char}' appear in this word?", str(char_counts[positive_char]))
    negative_char = random.choice(possible_negative_chars)
    
    questions = [
        (f"how many times does the character '{positive_char}' appear in this word?", str(char_counts[positive_char])),
        (f"how many times does the character '{negative_char}' appear in this word?", str(0))
    ]
    return random.choice(questions)

# =============================================================================
# BIG CATEGORY 2: POSITIONAL ANALYSIS
# =============================================================================

def generate_category_3_position(word):
    """Category 3: Character at specific position"""
    position = random.randint(1, len(word))
    return (f"what is the character at position {position} in this word?", word[position - 1])

def generate_category_4_relation(word):
    """Category 4: Character relative order/sequence"""
    if len(word) <= 1:
        # Fallback to recognition if word is too short
        return generate_category_0_recognition(word)
    
    ordered_pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
    positive_pair = random.choice(ordered_pairs)
    reversed_pair = (positive_pair[1], positive_pair[0])
    
    questions = [
        (f"does the character '{positive_pair[0]}' come before '{positive_pair[1]}' in this word?", 'yes'),
        (f"does the character '{reversed_pair[0]}' come before '{reversed_pair[1]}' in this word?", 'no')
    ]
    return random.choice(questions)

# =============================================================================
# BIG CATEGORY 3: STRUCTURAL ANALYSIS
# =============================================================================

def generate_category_5_structure(word):
    """Category 5: Word length/structure"""
    return ("what is the total number of characters in this word?", str(len(word)))

def generate_category_6_repeated_characters(word):
    """Category 6: Repeated character analysis"""
    # Count occurrences of each character
    char_counts = {}
    for char in word:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Check if any character appears more than once
    has_repeated = any(count > 1 for count in char_counts.values())
    
    if has_repeated:
        return ("is there any character that appears more than once in this word?", "yes")
    else:
        return ("is there any character that appears more than once in this word?", "no")

# =============================================================================
# BIG CATEGORY 4: BOUNDARY ANALYSIS
# =============================================================================

def generate_category_7_boundary_start(word):
    """Category 7: Word starting character"""
    actual_start_char = word[0]
    possible_negative_chars = list(set(chars) - {actual_start_char})
    if not possible_negative_chars:
        return (f"does this word start with the letter '{actual_start_char}'?", 'yes')
    negative_start_char = random.choice(possible_negative_chars)
    questions = [
        (f"does this word start with the letter '{actual_start_char}'?", 'yes'),
        (f"does this word start with the letter '{negative_start_char}'?", 'no')
    ]
    return random.choice(questions)

def generate_category_8_boundary_end(word):
    """Category 8: Word ending character"""
    actual_end_char = word[-1]
    possible_negative_chars = list(set(chars) - {actual_end_char})
    if not possible_negative_chars:
        return (f"does this word end with the letter '{actual_end_char}'?", 'yes')
    negative_end_char = random.choice(possible_negative_chars)
    questions = [
        (f"does this word end with the letter '{actual_end_char}'?", 'yes'),
        (f"does this word end with the letter '{negative_end_char}'?", 'no')
    ]
    return random.choice(questions)

# =============================================================================
# CATEGORY MAPPING - COMPATIBLE WITH EXISTING CODE
# =============================================================================

# Main category generators mapping (compatible with your existing code)
CATEGORY_GENERATORS = {
    0: generate_category_0_recognition,
    1: generate_category_1_existence,
    2: generate_category_2_quantity,
    3: generate_category_3_position,
    4: generate_category_4_relation,
    5: generate_category_5_structure,
    6: generate_category_6_repeated_characters,
    7: generate_category_7_boundary_start,
    8: generate_category_8_boundary_end
}

# Big category groupings for easy experimentation
BIG_CATEGORIES = {
    "CHARACTER_PRESENCE": [1, 2],       # Existence + quantity
    "POSITIONAL_ANALYSIS": [3, 4],      # Position + relation
    "STRUCTURAL_ANALYSIS": [5, 6],      # Length + repeated chars
    "BOUNDARY_ANALYSIS": [7, 8]         # Start + end
}

def generate_qas(word, categories):
    """
    Generates a list of (question, answer) pairs for a given word based on specified categories.
    
    Args:
        word (str): The input word.
        categories (list[int]): A list of category IDs (0-8) to generate questions for.
    
    Returns:
        list[tuple]: A list of (question, answer) tuples.
    """
    questions_answers = []
    for cat_id in categories:
        if cat_id in CATEGORY_GENERATORS:
            qa_pair = CATEGORY_GENERATORS[cat_id](word)
            questions_answers.append(qa_pair)
    
    return questions_answers

# =============================================================================
# HELPER FUNCTIONS FOR EXPERIMENTATION
# =============================================================================

def generate_qas_by_big_category(word, big_category_name):
    """
    Generate QAs for all subcategories within a big category.
    
    Args:
        word (str): The input word.
        big_category_name (str): One of "CHARACTER_PRESENCE", "POSITIONAL_ANALYSIS", 
                                "STRUCTURAL_ANALYSIS", "BOUNDARY_ANALYSIS"
    
    Returns:
        list[tuple]: A list of (question, answer) tuples.
    """
    if big_category_name not in BIG_CATEGORIES:
        raise ValueError(f"Unknown big category: {big_category_name}")
    
    categories = BIG_CATEGORIES[big_category_name]
    return generate_qas(word, categories)

def get_all_categories():
    """Return all available category IDs"""
    return list(CATEGORY_GENERATORS.keys())

def get_big_category_combinations():
    """Return useful category combinations for experiments"""
    return {
        "baseline_recognition": [0],
        "char_presence": [0, 1, 2],
        "positional": [0, 3, 4], 
        "structural": [0, 5, 6],
        "boundary": [0, 7, 8],
        "all_categories": list(range(9))
    }
