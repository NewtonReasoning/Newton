import pandas as pd
from query_gpt import ChatBot
import csv
import os

if __name__ == "__main__":
    # Load in csv file as pandas dataframe
    df = pd.read_csv('data/explicit_questions.csv')

    bot_gpt4 = ChatBot("Select the most correct answer letter from the provided options, no explanation. One word only.", model = 'gpt-4')
    bot_turbo = ChatBot("Select the most correct answer letter from the provided options, no explanation. One word only.", model = 'gpt-3.5-turbo')
    
    answer_file_prefix = 'explicit_questions_responses_gpt'
    # Initialize csv with headers
    if not os.path.exists(f'data/{answer_file_prefix}.csv'):
        with open(f'data/{answer_file_prefix}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['q_type','attribute','category','question','polarity', 'gt', 'choice_1', 'choice_2', 'choice_3', 'choice_4', 'gpt_turbo_response', 'gpt4_response'])
            
    # Iterate through the lines
    for index, row in df.iterrows():

        # Get the question and options
        question = row['question']
        option_1 = row['choice_1']
        option_2 = row['choice_2']
        option_3 = row['choice_3']
        option_4 = row['choice_4']

        question_type = row['q_type']


        # Format question template
        if question_type == 'boolean':
            question_template = f"Question: {question} A) {option_1} B) {option_2} Answer:"
        elif question_type == 'MC':
            question_template = f"Question: {question} A) {option_1} B) {option_2} C) {option_3} B) {option_4} Answer:"

        gpt_turbo_response = bot_turbo(question)
        gpt4_response = bot_gpt4(question)
        print(question_template)
        print('Ground Truth: ', row['gt'])
        print('GPT4: ', gpt4_response)
        print('GPT3.5: ',gpt_turbo_response)
        print('')

        row['gpt_turbo_response'] = gpt_turbo_response
        row['gpt4_response'] = gpt4_response

        # Save in a csv
        with open(f'data/{answer_file_prefix}.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([row['q_type'], row['attribute'], row['category'], row['question'],row['polarity'], row['gt'], row['choice_1'], row['choice_2'], row['choice_3'], row['choice_4'], gpt_turbo_response.lower().replace('.',''), gpt4_response.lower().replace('.','')])