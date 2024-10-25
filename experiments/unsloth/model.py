from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

class LLaMaUnSloth:
    def __init__(self, max_seq_length=2048, dtype=None, load_in_4bit=True):
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit 
        self.load_model()

    def load_model(self):
        self.model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        self.tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference
    def generate (self,question:str, answer:str)->str:
        
        template = \
        f"""Consider the following statement as an answer to the question "{question}". 
        Statement: "{answer}"
        Based on this statement, does it indicate "yes" or "no" in response to the question "{question}"? 
        Please respond with only "yes" or "no"."""
        messages = [
            {"role": "user", "content": template},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")


        text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        k = self.model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                        use_cache = True, temperature = 1.5, min_p = 0.1)
        output_text = self.tokenizer.decode(k[0], skip_special_tokens=True)
        response = output_text.split("\n\n")[-1].strip().strip(".")  # 
        return response
