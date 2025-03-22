import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Use a T5-based legal model for text generation
MODEL_NAME = "SEBIS/legal_t5_small_summ_en"  # Change if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_legal_response(user_input):
    """Generates a structured response using LegalT5."""
    
    prompt = f"Legal Advice: {user_input}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response_text

# Example usage
query = "I was cheated in a business deal, and the person is not returning my money."
legal_response = generate_legal_response(query)
print("\n🔹 Legal Advice:\n", legal_response)
