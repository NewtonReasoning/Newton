from transformers import pipeline, set_seed
import torch
import argparse
import pandas as pd
import os
import csv
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM

set_seed(42)


class unifiedqa:
    def __init__(self, model_name):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda:0")

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to("cuda:0")
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def get_output(self, prompt):
        output = self.run_model(prompt, max_length=128, do_sample=False)[0]
        return output

class dolly:
    def __init__(self, model_name):
        self.model = pipeline(model=model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    def get_output(self, prompt):
        output = self.model(prompt)
        return output[0]["generated_text"]    

class flan:
    def __init__(self,model_name):
        self.model = pipeline(model=model_name, device_map="auto")
    def get_output(self, prompt):
        output = self.model(prompt, max_length=128, do_sample=False)
        return output[0]["generated_text"]

class alpaca:
    def __init__(self,model_name):
        self.tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
        self.model = LlamaForCausalLM.from_pretrained(
            "chainyo/alpaca-lora-7b",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
        )

        self.model.eval()
        if torch.__version__ >= "2":
            self.model = torch.compile(self.model)

                
    def generate_prompt(self,instruction: str, input_ctxt: str = None) -> str:
        if input_ctxt:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input_ctxt}

    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:"""
    def get_output(self, prompt):
        instruction = prompt
        input_ctxt = None  # For some tasks, you can provide an input context to help the model generate a better response.

        prompt = self.generate_prompt(instruction, input_ctxt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return response

                

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Huggingface Inference Args')
    parser.add_argument("-m", "--model", required=True,help = "model = model name")
    parser.add_argument("-t", "--track", required=True,help = "track=['foundational','explicit','implicit']")

    args = parser.parse_args()

    if args.track == 'foundational':
        raise ValueError('Foundational track not supported')
    elif args.track == 'explicit':
        raise ValueError('Explicit track not supported')
    elif args.track == 'implicit':
        df = pd.read_csv('data/implicit_questions.csv')

    models = [args.model]
    rows = df.to_dict('records')

    for model_name in models:
        if '/' in model_name:
            answer_file_prefix = f'{args.track}_questions_responses_{model_name.split("/")[1]}'
        else:
            answer_file_prefix = f'{args.track}_questions_responses_{model_name}'

        if args.track == 'explicit':
            if not os.path.exists(f'data/{answer_file_prefix}.csv'):
                with open(f'data/{answer_file_prefix}.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['q_type','attribute','category','question','polarity', 'gt', 'choice_1', 'choice_2', 'choice_3', 'choice_4', f'{model_name}_response'])
        elif args.track == 'implicit':
            if not os.path.exists(f'data/{answer_file_prefix}.csv'):
                with open(f'data/{answer_file_prefix}.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['q_type','tag','context', 'question','gt', 'choice_1', 'choice_2', 'choice_3', 'choice_4', f'{model_name}_response'])
        # Load the model
        if 'flan' in model_name:
            model = flan(model_name)
        elif 'unifiedqa' in model_name:
            model = unifiedqa(model_name)
        elif 'dolly' in model_name:
            model = dolly(model_name)
        elif 'alpaca' in model_name:
            model = alpaca(model_name)


        for index, row in enumerate(rows):
            
            context = row['context']
            question = row['question']
            candidate1 = row['choice_1']
            candidate2 = row['choice_2']
            candidate3 = row['choice_3']
            candidate4 = row['choice_4']

            if model_name == 'allenai/unifiedqa-v2-t5-large-1363200':
                prompt = f'Context: {context} \\n Question: {question} \\n (a) {candidate1} (b) {candidate2} (c) {candidate3} (d) {candidate4}'
            else:
                prompt = f'Context: {context} \\n Question: {question} \\n (a) {candidate1} (b) {candidate2} (c) {candidate3} (d) {candidate4}'

            prediction = model.get_output(prompt)

            print(model_name)
            print(prompt)
            print(row['gt'])
            print(prediction)

            # Save in a csv file
            if args.track == 'explicit':
                with open(f'data/{answer_file_prefix}.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([row['q_type'], row['attribute'], row['category'], row['question'],row['polarity'], row['gt'], row['choice_1'], row['choice_2'], row['choice_3'], row['choice_4'], prediction])
            elif args.track == 'implicit':
                with open(f'data/{answer_file_prefix}.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([row['q_type'], row['tag'], row['context'], row['question'], row['gt'], row['choice_1'], row['choice_2'], row['choice_3'], row['choice_4'], prediction])