import pandas as pd
from itertools import product
import random
import itertools



def filter_categories(df, criteria_dict):
    # Create a boolean mask to filter rows based on the criteria dictionary
    mask = pd.Series(True, index=df.index)  # Initialize with True values for all rows

    for attribute, value in criteria_dict.items():
        # Determine the result_majority value based on the criteria
        result_majority_value = 3.0 if value == 'high' else 1.0

        # Update the boolean mask based on the criteria for the current attribute
        mask &= (df['attribute'] == attribute) & (df['result_majority'] == result_majority_value)

    # Filter the DataFrame based on the boolean mask and extract the unique categories
    filtered_df = df[mask]
    categories = filtered_df['category'].unique().tolist()

    return categories

def _generate_combinations(list1, list2):
    combinations = [sublist1 + sublist2 for sublist1, sublist2 in product(list1, list2)]
    return combinations

def generate_combinations(positives, negatives, p, n, max_samples = None,seed=42):
    random.seed(seed)
    combinations_list = []

    # Generate all possible combinations of p positive objects
    positive_combinations = list(itertools.combinations(positives, p))

    print('len positive_combinations', len(positive_combinations))


    # Generate all possible combinations of n negative objects
    print('len neg', len(negatives))
    negative_combinations = list(itertools.combinations(negatives, n))
    print('len negative_combinations',len(negative_combinations))

    if max_samples:

        # num_negative_samples = max_samples / len(positive_combinations)
        if len(negative_combinations) > max_samples:
            negative_combinations = random.sample(negative_combinations, max_samples)


            for index, negative_comb in enumerate(negative_combinations):
                true_idx = index % len(positive_combinations)
                positive_comb = positive_combinations[true_idx]
                combined_list = list(positive_comb) + list(negative_comb)
                combinations_list.append(combined_list)
        else:
            combinations_list = _generate_combinations(positive_combinations, negative_combinations)
            if len(combinations_list) > max_samples:
                combinations_list = random.sample(combinations_list, max_samples)
                
    return combinations_list