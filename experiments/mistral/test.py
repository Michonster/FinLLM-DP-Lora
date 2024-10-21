import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import time

def compute_f1(TP, FP, FN):
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def compute_prediction(predictions, references):
    TP = FP = FN = 0
    for pred, ref in zip(predictions, references):
        for p, r in zip(pred, ref):
            if p == 1 and r == 1:
                TP += 1
            elif p == 1 and r == 0:
                FP += 1
            elif p == 0 and r == 1:
                FN += 1
    return TP, FP, FN

TOKEN = "your-token-here"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=TOKEN).to(device) 
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=TOKEN)

dataset = load_dataset("TheFinAI/flare-cd", token=TOKEN)
dataset_iter = dataset['test'].iter(1)

if tokenizer.pad_token is None:
    print("Setting pad_token to eos_token")
    tokenizer.pad_token = tokenizer.eos_token

start_time = time.time()
message = [{"role": "user", "content": ""}]
try:
    while True:
        data = next(dataset_iter)
        query = data['query'][0]
        message[0]["content"] = query

        encodeds = tokenizer.apply_chat_template(message, return_tensors="pt", padding="max_length", max_length=512, return_attention_mask=True)
        model_inputs = encodeds.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(decoded[0])

except StopIteration:
    end_time = time.time()
    print("Processing complete.")
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"An error occurred: {e}")
