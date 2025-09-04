# pip install peft transformers bitsandbytes accelerate datasets

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"

base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
lora_model = PeftModel.from_pretrained(base_model, "mistral-hr-lora")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

prompt = "What's the policy on bereavement leave?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
output = lora_model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
