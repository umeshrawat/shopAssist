# generator_phi.py
class LocalGenerator:
    def __init__(self, model_id="microsoft/phi-2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt, max_tokens=300, temperature=0.7):
        output = self.pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
        return output[0]['generated_text'].replace(prompt, '').strip()
