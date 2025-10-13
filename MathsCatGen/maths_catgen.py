import pandas as pd
import re
from typing import List, Tuple, Optional, Dict, Any
import random


def get_maths_tasks():
    """
    These are mathematical tasks that we test the models on 
    """
    return [
        "minimum",
        "maximum", 
        "sum",
        "difference",
        "product",
        "average",
        "exponential" # Excluded from model evaluation for now 
    ]

def get_prompt_template():
    """
    Returns the prompt template for a given mathematical task.
    """
    return "Answer minimally: Given the numbers {x} and {y} calculate the {task}"


def is_ground_truth_correct(answer: str, ground_truth: str) -> bool:
    """
    Returns True if the ground_truth appears as the final number in the answer, ignoring whitespace and punctuation.
    Accepts answers like '13', '13.', '13**', 'The answer is 13', '**13**', 'random text **13** random text', 'boxed{13}'.
    """

    # Remove trailing whitespace and punctuation
    answer_clean = answer.strip().rstrip('.!**')
    # Find all numbers in the answer (including negative numbers and those with commas)
    numbers = re.findall(r'-?[\d,]+', answer_clean)
    
    # Remove commas from the numbers for comparison
    numbers_clean = [num.replace(',', '') for num in numbers]

    answer_no_comma = answer.replace(",", "")

    return (ground_truth == answer_no_comma or
            "**"+ground_truth+"**" in answer or
            "boxed{"+ground_truth+"}" in answer or
            ""+ground_truth+" " in answer_no_comma  or
            ""+ground_truth+"." in answer_no_comma  or
            # Check that the last number matches within 0.001 tolerance
            (numbers_clean and abs(float(numbers_clean[-1]) - float(ground_truth)) < 0.001))

def test_is_ground_truth_correct():
    assert is_ground_truth_correct("-14338", "-14338")
    assert is_ground_truth_correct("-14338 ", "-14338")
    assert is_ground_truth_correct("-14338.", "-14338")
    assert is_ground_truth_correct("-14,338", "-14338")
    assert is_ground_truth_correct("-14,338 ", "-14338")
    assert is_ground_truth_correct("-14,338.", "-14338")
    assert not is_ground_truth_correct("-14339", "-14338")
    assert is_ground_truth_correct("boxed{-14338}", "-14338")
    assert is_ground_truth_correct("**-14338**", "-14338")
    assert is_ground_truth_correct("blah blah-14338 blah blah", "-14338")
    assert is_ground_truth_correct("135,702,468 - 269,485,731 = **-133,783,263**", "-133783263")
    assert is_ground_truth_correct("12123 - 12312 = -14338", "-14338")

def generate_number_pairs(n_examples: int = 200, 
                         min_val: int = 1, 
                         max_val: int = 99,
                         include_negatives: bool = False,
                         seed: int = 42) -> List[Tuple[int, int]]:
    """Generate diverse number pairs for testing"""
    random.seed(seed)
    pairs = []
    
    # Strategy: Mix of different number ranges for variety
    ranges = [
        (1, 9),      # Single digits
        (10, 99),    # Double digits
        (1, 99),     # Mixed
    ]
    
    if include_negatives:
        ranges.extend([
            (-99, -1),   # Negative numbers
            (-50, 50),   # Mixed positive/negative
        ])
    
    examples_per_range = n_examples // len(ranges)
    
    for min_r, max_r in ranges:
        for _ in range(examples_per_range):
            x = random.randint(min_r, max_r)
            y = random.randint(min_r, max_r)
            pairs.append((x, y))
    
    # Fill remaining with random pairs from full range
    while len(pairs) < n_examples:
        x = random.randint(min_val, max_val)
        y = random.randint(min_val, max_val)
        pairs.append((x, y))
    
    random.shuffle(pairs)
    return pairs[:n_examples]

def calculate_ground_truth(x: int, y: int, operation: str) -> str:
    """Calculate the correct answer for a given operation"""
    if operation == "minimum":
        return str(min(x, y))
    elif operation == "maximum":
        return str(max(x, y))
    elif operation == "sum":
        return str(x + y)
    elif operation == "difference":
        return str(abs(x - y))  # Assuming absolute difference
    elif operation == "product":
        return str(x * y)
    elif operation == "average":
        return str((x + y) / 2)
    elif operation == "exponential":
        # Limit exponential to prevent overflow
        try:
            result = x ** y
            # Cap at reasonable size
            if result > 10**15:
                return "OVERFLOW"
            return str(result)
        except:
            return "OVERFLOW"
    else:
        raise ValueError(f"Unknown operation: {operation}")

def generate_synthetic_data(tasks, prompt_template, n_examples_per_task: int = 200) -> pd.DataFrame:
    """Generate synthetic data for all tasks"""
    
    all_data = []
    
    for task in tasks:
        
        # For exponential, use smaller Y values to prevent overflow
        if task == "exponential":
            pairs = generate_number_pairs(n_examples_per_task, min_val=2, max_val=15)
            # Limit Y further for exponential
            pairs = [(x, min(y, 10)) for x, y in pairs]
        else:
            pairs = generate_number_pairs(n_examples_per_task)
        
        for x, y in pairs:
            prompt = prompt_template.format(x=x, y=y, task=task)
            ground_truth = calculate_ground_truth(x, y, task)
            
            # Skip overflow cases
            if ground_truth == "OVERFLOW":
                continue
                
            all_data.append({
                "task": task,
                "x": x,
                "y": y,
                "prompt": prompt,
                "ground_truth": ground_truth
            })
    
    df = pd.DataFrame(all_data)
    print(f"\nGenerated {len(df)} total examples across {len(tasks)} tasks")
    print(f"Examples per task: {df['task'].value_counts().to_dict()}")
    
    return df

