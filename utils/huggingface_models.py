from transformers import pipeline, set_seed
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

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
    
class falcon:
    def __init__(self,model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    def get_output(self, prompt):

        sequences = self.pipeline(
        prompt,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return sequences[0]["generated_text"]

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
        # print(response)
        return response