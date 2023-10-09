import random 
from utils.filter_generate import filter_categories, generate_combinations
import pandas as pd
import os
import csv

def get_inversed_attributes(attributes):
    for attribute in attributes.keys():
        if attributes[attribute] == 'high':
            attributes[attribute] = 'low'
        elif attributes[attribute] == 'low':
            attributes[attribute] = 'high'
    return attributes

if __name__=='__main__': 
    gen_boolean_questions_flag = True
    gen_mc_questions_flag = True

    # Load in the csv file as a pandas dataframe
    filtered_df = pd.read_csv('data/confident_questions.csv')

    if not os.path.exists('data/implicit_questions.csv'):
        with open('data/implicit_questions.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['q_type','tag','context', 'question','gt', 'choice_1', 'choice_2', 'choice_3', 'choice_4'])

    
    if gen_mc_questions_flag:
        print('------------------Generating multiple choice questions--------------------')

        questions_total = 0
        question_type = 'MC'

        templates = [{'Context': 'I am packing a backpack.', 
                      'question': 'Which of <object 1>, <object 2>, <object 3>, <object 4> should I put at the bottom?',
                      'attribute': {'Stiffness': 'high', 'Brittleness': 'low'},
                      'category': 'arrangement',
                      'tag': 'backpack_packing'},
                      {'Context': 'I have an irregularly shaped space in my suitcase, and four objects with the same volume.', 
                      'question': 'Which of <object 1>, <object 2>, <object 3>, <object 4> could fit in that space?',
                      'attribute': {'Malleability': 'high', 'Elasticity': 'high'},
                      'category': 'arrangement',
                      'tag': 'suitcase_packing'},
                      {'Context': 'I need an object to place sandpaper above.', 
                      'question': 'Which of <object 1>, <object 2>, <object 3>, <object 4> is the most suitable?',
                      'attribute': {'Surface Hardness': 'high', 'Surface smoothness': 'low'},
                      'category': 'arrangement',
                      'tag': 'sandpaper_placing'},
                      {'Context': 'I am trying to open some plastic packaging.', 
                      'question': 'Which of <object 1>, <object 2>, <object 3>, <object 4> can help me open?',
                      'attribute': {'Sharpness': 'high'},
                      'category': 'tool_use',
                      'tag': 'plastic_packaging'},
                      {'Context': 'I am wrapping fragile gifts and want to protect them from impact.', 
                      'question': 'Which of <object 1>, <object 2>, <object 3>, <object 4> should I choose for cushioning?',
                      'attribute': {'Softness': 'high', 'Elasticity': 'high'},
                      'category': 'tool_use',
                      'tag': 'fragile_gifts'},
                      {'Context': 'I need to hammer a nail into a solid wooden board.', 
                      'question': 'Which of <object 1>, <object 2>, <object 3>, <object 4> should I choose?',
                      'attribute': {'Surface Hardness': 'high', 'Brittleness': 'low', 'Stiffness': 'high'},
                      'category': 'tool_use',
                      'tag': 'hammering'},
                      {'Context': 'I need to prepare a work table as a play area for kids.', 
                      'question': 'Which of <object 1>, <object 2>, <object 3>, <object 4> should I remove?',
                      'attribute': {'Sharpness': 'high'},
                      'category': 'safety',
                      'tag': 'kids_play_area'},
                      {'Context': 'I have a robot which sorts objects by tossing them to bins.', 
                      'question': 'Which of <object 1>, <object 2>, <object 3>, <object 4> can be tossed safely?',
                      'attribute': {'Brittleness': 'low', 'Elasticity': 'high'},
                      'category': 'safety',
                      'tag': 'robot_tossing'},
                      {'Context': 'I need an object to provide insulation for a sharp edge on a piece of furniture.',
                       'question': 'Which of <object 1>, <object 2>, <object 3>, <object 4> should I choose?',
                       'attribute': {'Softness': 'high', 'Elasticity': 'high'},
                        'category': 'safety',
                        'tag': 'insulation'},
                        ]
  
        for template in templates:
            print(f'------------------{template["tag"]}--------------------')

            context = template['Context']
            question = template['question']
            attributes = template['attribute']
            q_category = template['category']
            tag = template['tag']

            positives = filter_categories(df = filtered_df, criteria_dict = attributes)
            inv_attributes = get_inversed_attributes(attributes)
            negatives = filter_categories(df = filtered_df, criteria_dict = inv_attributes)
            print(positives)
            print(negatives)

            combinations_list = generate_combinations(positives, negatives, p=1, n=3, max_samples=10000)
            print('Combinations list: ', len(combinations_list))
            print('Positives: ', len(positives))
            print('Negatives: ', len(negatives))
            for combination in combinations_list:
                combination = list(combination)
                gt = combination[0]
                
                new_question = question
                random.shuffle(combination)
                choice_1 = combination[0]
                choice_2 = combination[1]
                choice_3 = combination[2]
                choice_4 = combination[3]
                new_question = new_question.replace('<object 1>', combination[0])
                new_question = new_question.replace('<object 2>', combination[1])
                new_question = new_question.replace('<object 3>', combination[2])
                new_question = new_question.replace('<object 4>', combination[3])

                # Write to csv
                with open('data/implicit_questions.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([question_type,tag, context, new_question, gt, choice_1, choice_2, choice_3, choice_4])
                questions_total += 1
