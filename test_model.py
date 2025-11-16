from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model + tokenizer
model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.float16
)

# Test prompt // This comment out of i
prompt = "I have tomato crops suffering from early blight. Suggest 5 steps including pesticide use and organic solutions."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
