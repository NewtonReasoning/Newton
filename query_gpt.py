import openai
openai.organization = "INSERT ORGANIZATION"
openai.api_key = 'INSERT API KEY'
openai.Model.list()
import time

import retry

class ChatBot:
    def __init__(self, system="", model = "gpt-3.5-turbo"):
        self.system = system
        self.messages = []
        self.model = model
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        self.messages_to_execute = self.messages + [{"role": "user", "content": message}]
        result = self.execute()
        self.messages_to_execute = []
        return result
    
    @retry.retry(tries=5, delay=5, max_delay=10)
    def execute(self):
        completion = openai.ChatCompletion.create(model=self.model, messages=self.messages_to_execute,top_p = 0.5, timeout=10)
        return completion.choices[0].message.content

class ChatBotOld:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self):
        try:
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages,top_p = 0.1)
            return completion.choices[0].message.content
        except openai.error.APIConnectionError as e:
            print(f"APIConnectionError: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            return self.execute()