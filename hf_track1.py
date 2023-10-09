# import transformers
from transformers import AutoTokenizer, set_seed, enable_full_determinism
# transformers.set_seed(0)

import torch
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
from transformers import AutoModelForMultipleChoice
import pandas as pd
import os
import csv

set_seed(42)
enable_full_determinism(42)

import sys
sys.path.append('..')
from alpaca_run import unifiedqa, dolly, flan, falcon, alpaca

if __name__ == "__main__":

    df = pd.read_csv('data/confident_questions.csv')

    prompt = "Brittleness: How easy is it for a belt to break (shatter, crack) when a sudden impact force is applied?"
    candidate1 = "Low: object can withstand most impact forces (drop, smash, etc.)."
    candidate2 = "Moderate: object can withstand minor impact forces."
    candidate3 = "High: object shatters easily with impact force."
    models = ["declare-lab/flan-alpaca-gpt4-xl",'google/flan-t5-large',
              'google/flan-t5-xl', 'google/flan-t5-small',
              'google/flan-t5-base','databricks/dolly-v2-3b', 'databricks/dolly-v2-7b', 
              'chainyo/alpaca-lora-7b', "allenai/unifiedqa-v2-t5-large-1363200"]

    rows = df.to_dict('records')

    for model_name in models:
        if '/' in model_name:
            answer_file_prefix = f'confident_questions_responses_{model_name.split("/")[1]}'
        else:
            answer_file_prefix = f'confident_questions_responses_{model_name}'


        if not os.path.exists(f'data/{answer_file_prefix}.csv'):
            with open(f'data/{answer_file_prefix}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['attribute', 'category', 'type', 'question', 'option_1', 'option_2', 'option_3','result_majority', 'agreement', f'{model_name}_response'])
        
        # Load the model
        if 'flan' in model_name:
            model = flan(model_name)
        elif 'unifiedqa' in model_name:
            model = unifiedqa(model_name)
        elif 'dolly' in model_name:
            model = dolly(model_name)
        elif 'alpaca' in model_name:
            model = alpaca(model_name)
        elif 'falcon' in model_name:
            model = falcon(model_name)

        for index, row in enumerate(rows):

            question = row['question']
            candidate1 = row['option_1']
            candidate2 = row['option_2']
            candidate3 = row['option_3']

            prompt = prompt = f'{question} \\n (a) {candidate1} (b) {candidate2} (c) {candidate3}'


            prediction = model.get_output(prompt)

            print(model_name)
            print(prompt)
            print(row['result_majority'])
            print(prediction)

            # Save in a csv
            with open(f'data/{answer_file_prefix}.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([row['attribute'], row['category'], row['type'], row['question'], row['option_1'], row['option_2'], row['option_3'], row['result_majority'], row['agreement'], prediction])
