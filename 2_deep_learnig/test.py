import numpy as np

def solution(A, B):
    max_possible = min(A, B)
    left = 1
    right = max_possible
    best = 0
    
    while left <= right:
        mid = (left + right) // 2
        total_sticks = (A // mid) + (B // mid)
        if total_sticks >= 4:
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best

def solution(A, B):
    # 上界设为 (A+B) // 4，因为 4 根棍子的总长度至少为 4*L
    max_possible = (A + B) // 4  
    left = 1
    right = max_possible
    best = 0
    
    while left <= right:
        mid = (left + right) // 2
        total_sticks = (A // mid) + (B // mid)
        if total_sticks >= 4:
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best

print(solution(1, 8))  # 输出应为 2

def solution(letters):
    count = 0
    for c in 'abcdefghijklmnopqrstuvwxyz':
        lower = c
        upper = c.upper()
        if lower in letters and upper in letters:
            last_lower = letters.rfind(lower)
            first_upper = letters.find(upper)
            if last_lower < first_upper:
                count += 1
    return count

def solution(letters):
    # 如果 letters 不是字符串，则尝试将其转换为字符串
    if not isinstance(letters, str):
        letters = ''.join(letters)
    
    count = 0
    for c in 'abcdefghijklmnopqrstuvwxyz':
        lower = c
        upper = c.upper()
        if lower in letters and upper in letters:
            last_lower = letters.rfind(lower)
            first_upper = letters.find(upper)
            if last_lower < first_upper:
                count += 1
    return count

def solution(letters):
    """
    Counts the number of different letters that appear in both lowercase and uppercase,
    where all lowercase occurrences appear before any uppercase occurrence.

    Args:
        letters (str): The input string of English letters.

    Returns:
        int: The number of different letters fulfilling the conditions.
    """
    seen_lower = set()  # Set to track lowercase letters seen
    invalid_letters = set()  # Set to track letters that violate the condition
    valid_letters = set()  # Set to track valid letters

    for char in letters:
        if char.islower():
            # If it's a lowercase letter, add it to the seen_lower set
            if char not in invalid_letters:
                seen_lower.add(char)
        elif char.isupper():
            # If it's an uppercase letter
            lower_char = char.lower()
            if lower_char not in seen_lower:
                # If the corresponding lowercase letter has not been seen, it's invalid
                invalid_letters.add(lower_char)
                if lower_char in valid_letters:
                    valid_letters.remove(lower_char)  # Remove from valid if previously added
            else:
                # If the lowercase letter was seen before, it's valid
                if lower_char not in invalid_letters:
                    valid_letters.add(lower_char)

    # Return the count of valid letters
    return len(valid_letters)

# Example usage
letters = "ABCabcAefG"
print(solution(letters))  # Output: 0

def weighted_f1(y_true, y_pred, f1_weights):
    classes = set(y_true).union(set(y_pred))
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    
    for cls in classes:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for true, pred in zip(y_true, y_pred):
            if true == cls and pred == cls:
                true_positives += 1
            elif pred == cls and true != cls:
                false_positives += 1
            elif true == cls and pred != cls:
                false_negatives += 1
        
        # Calculate precision
        precision_denominator = true_positives + false_positives
        precision = true_positives / precision_denominator if precision_denominator != 0 else 0
        precision_dict[cls] = precision
        
        # Calculate recall
        recall_denominator = true_positives + false_negatives
        recall = true_positives / recall_denominator if recall_denominator != 0 else 0
        recall_dict[cls] = recall
        
        # Calculate F1 score
        f1_numerator = 2 * precision * recall
        f1_denominator = precision + recall if (precision + recall) != 0 else 1
        f1_score = f1_numerator / f1_denominator
        f1_dict[cls] = f1_score
    
    # Calculate weighted F1
    weighted_f1_score = 0
    for cls in f1_weights:
        if cls in f1_dict:
            weighted_f1_score += f1_dict[cls] * f1_weights[cls]
    
    return {
        'precision': precision_dict,
        'recall': recall_dict,
        'F1': f1_dict,
        'weighted_F1': weighted_f1_score
    }