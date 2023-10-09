import random 
from utils.filter_generate import filter_categories, generate_combinations
import pandas as pd
import os
import csv

if __name__=='__main__': 
    gen_boolean_questions_flag = True
    gen_mc_questions_flag = True

    # Load in the csv file as a pandas dataframe
    filtered_df = pd.read_csv('data/confident_questions.csv')
    attributes = ['Surface Hardness', 'Brittleness', 'Softness', 'Sharpness', 'Stiffness', 'Malleability', 'Elasticity', 'Surface smoothness']
    attributes_renamed = {'Elasticity': 'elastic',
                                  'Surface Hardness': 'hard on the surface',
                                  'Softness': 'soft',
                                  'Sharpness': 'sharp',
                                  'Stiffness': 'stiff', 
                                  'Malleability': 'malleable',
                                  'Brittleness': 'brittle',
                                  'Surface smoothness': 'smooth'}
    
    # Initialize csv with headers
    if not os.path.exists('data/explicit_questions.csv'):
        with open('data/explicit_questions.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['q_type','attribute','category','question','polarity', 'gt', 'choice_1', 'choice_2', 'choice_3', 'choice_4'])

    if gen_boolean_questions_flag:

        question_type = 'boolean'
        questions_total = 0

        for attribute in attributes:
            print(attribute)
            positives = filter_categories(df = filtered_df, criteria_dict = {f'{attribute}': 'high'})
            negatives = filter_categories(df = filtered_df, criteria_dict = {f'{attribute}': 'low'})
            
            combinations_list = generate_combinations(positives, negatives, p=1, n=1, max_samples=1200)
            print('Combinations list: ', len(combinations_list))
            print('Positives: ', len(positives))
            print('Negatives: ', len(negatives))
            for combination in combinations_list:
                template_1_true = f"{combination[0]} is more {attributes_renamed[attribute]} than {combination[1]}."
                template_1_false = f"{combination[1]} is more {attributes_renamed[attribute]} than {combination[0]}."
                template_2_false = f"{combination[0]} is less {attributes_renamed[attribute]} than {combination[1]}."
                template_2_true = f"{combination[1]} is less {attributes_renamed[attribute]} than {combination[0]}."
                questions_total += 4
            
                # Write to csv
                with open('data/explicit_questions.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([question_type, attribute, combination[0], template_1_true, 'pos', True, True, False, 'N/A', 'N/A'])
                    writer.writerow([question_type,attribute, combination[0], template_1_false, 'pos', False, True, False, 'N/A', 'N/A'])
                    writer.writerow([question_type,attribute, combination[0], template_2_true, 'neg', True, True, False , 'N/A', 'N/A'])
                    writer.writerow([question_type,attribute, combination[0], template_2_false, 'neg', False, True, False, 'N/A', 'N/A'])
            for positive in positives:

                template_3_true = f"{positive} is {attributes_renamed[attribute]}."
                template_4_false = f"{positive} is not {attributes_renamed[attribute]}."
                # Write to csv
                with open('data/explicit_questions.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([question_type,attribute, positive, template_3_true, 'pos', True, True, False, 'N/A', 'N/A'])
                    writer.writerow([question_type,attribute, positive, template_4_false, 'neg', False, True, False, 'N/A', 'N/A'])
                questions_total += 2
            for negative in negatives:
                template_3_false = f"{negative} is {attributes_renamed[attribute]}."
                template_4_true = f"{negative} is not {attributes_renamed[attribute]}."

                # Write to csv
                with open('data/explicit_questions.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([question_type,attribute, negative, template_3_false, 'pos', False, True, False, 'N/A', 'N/A'])
                    writer.writerow([question_type,attribute, negative, template_4_true, 'neg', True, True, False, 'N/A', 'N/A'])
                questions_total += 2
        print(questions_total)

    
    if gen_mc_questions_flag:
        print('------------------Generating multiple choice questions--------------------')

        questions_total = 0
        question_type = 'MC'

        for attribute in attributes:
            print(attribute)
            positives = filter_categories(df = filtered_df, criteria_dict = {f'{attribute}': 'high'})
            negatives = filter_categories(df = filtered_df, criteria_dict = {f'{attribute}': 'low'})
            
            combinations_list = generate_combinations(positives, negatives, p=1, n=3, max_samples=2500)
            print('Combinations list: ', len(combinations_list))
            print('Positives: ', len(positives))
            print('Negatives: ', len(negatives))
            for combination in combinations_list:
                combination = list(combination)
                gt = combination[0]
                random.shuffle(combination)
                template_1 = f"Which object is the most {attributes_renamed[attribute]}?"
                choice_1 = combination[0]
                choice_2 = combination[1]
                choice_3 = combination[2]
                choice_4 = combination[3]

                # Write to csv
                with open('data/explicit_questions.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([question_type,attribute, gt, template_1, 'pos', gt, choice_1, choice_2, choice_3, choice_4])

            combinations_list = generate_combinations(negatives, positives, p=1, n=3, max_samples=2500)
            print('Combinations list: ', len(combinations_list))
            print('Positives: ', len(negatives))
            print('Negatives: ', len(positives))
            for combination in combinations_list:
                combination = list(combination)
                gt = combination[0]
                random.shuffle(combination)
                template_1 = f"Which object is the least {attributes_renamed[attribute]}?"
                choice_1 = combination[0]
                choice_2 = combination[1]
                choice_3 = combination[2]
                choice_4 = combination[3]

                # Write to csv
                with open('data/explicit_questions.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([question_type,attribute, gt, template_1, 'neg', gt, choice_1, choice_2, choice_3, choice_4])