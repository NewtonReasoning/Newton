import pandas as pd
from query_gpt import ChatBot
import csv
import os


if __name__ == "__main__":
    # Load in csv file as pandas dataframe
    df = pd.read_csv('data/confident_questions.csv')

    bot_gpt4 = ChatBot("Select the most correct answer letter from the provided options, no explanation. One word only.", model = 'gpt-4')
    bot_turbo = ChatBot("Select the most correct answer letter from the provided options, no explanation. One word only.", model = 'gpt-3.5-turbo')
    
    answer_file_prefix = 'confident_questions_responses_4'

    # Initialize csv with headers
    if not os.path.exists(f'data/{answer_file_prefix}.csv'):
        with open(f'data/{answer_file_prefix}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['attribute', 'category', 'type', 'question', 'option_1', 'option_2', 'option_3','result_majority', 'agreement', 'gpt_turbo_response', 'gpt4_response'])
            
    # Iterate through the lines
    for index, row in df.iterrows():
        # Get the question and options
        question = row['question']
        option_1 = row['option_1']
        option_2 = row['option_2']
        option_3 = row['option_3']

        # Format question template
        question_template = f"Question: {question} A) {option_1} B) {option_2} C) {option_3} Answer:"

        gpt_turbo_response = bot_turbo(question)
        gpt4_response = bot_gpt4(question)
        print(question_template)
        print('GPT4: ', gpt4_response)
        print('GPT3.5: ',gpt_turbo_response)
        print('')

        row['gpt_turbo_response'] = gpt_turbo_response
        row['gpt4_response'] = gpt4_response
        
        # Save in a csv
        with open(f'data/{answer_file_prefix}.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([row['attribute'], row['category'], row['type'], row['question'], row['option_1'], row['option_2'], row['option_3'], row['result_majority'], row['agreement'], gpt_turbo_response.lower().replace('.',''), gpt4_response.lower().replace('.','')])