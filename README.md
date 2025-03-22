# NyaySaarthi
# 🏛️ Legal AI - Indian Law Q&A Model

This project trains a **Legal AI model** to answer legal queries based on **Indian constitutional laws**, **IPC sections**, and **NALSA (National Legal Services Authority) guidelines**. The model is fine-tuned using **GPT-2/InLegalBERT** and trained on a dataset extracted from legal PDFs.

## 📌 Features
- 📖 **Understands Indian Laws** - Answers questions based on IPC, Constitution, and NALSA guidelines.
- 📜 **PDF-Based Legal Data** - Extracts and processes text from official legal documents.
- ⚡ **Fine-tuned on Legal Texts** - Uses real legal texts for accuracy.
- 🤖 **AI-powered Chatbot** - Can be deployed as a chatbot or legal assistant.
- 📂 **Structured Dataset** - Converts unstructured PDFs into a machine-readable format.

## 📂 Dataset
The dataset is created by extracting text from **NALSA guidelines PDF** and **IPC legal documents**.

- **Sources:**
  - **NALSA Legal Aid Guidelines (PDF)**
  - **Indian Penal Code (IPC) (PDF)**
- **Preprocessing:**
  - Extracted text from PDFs using PyMuPDF (fitz)
  - Cleaned and structured data for training
  
## 🚀 Installation
Ensure you have Python **3.8+** installed, then run:

```bash
pip install transformers datasets torch pandas pypdf
```

## 🔧 Preprocessing PDF Data
Convert legal PDFs into structured text:

```python
import fitz  # PyMuPDF
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text("text") for page in doc])
    return text

# Load PDFs
nalsa_text = extract_text_from_pdf("nalsa_guidelines.pdf")
ipc_text = extract_text_from_pdf("ipc_laws.pdf")

# Create dataset
df = pd.DataFrame({"text": [nalsa_text, ipc_text]})
dataset = Dataset.from_pandas(df)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True, remove_columns=["text"])
```

## 🎯 Training the Model
Fine-tune **GPT-2** or **InLegalBERT**:

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("gpt2")
training_args = TrainingArguments(
    output_dir="./legal_gpt_results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
```

## 🧪 Testing the Model
```python
def generate_legal_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_legal_answer("What is Article 21?"))
```

## 🚀 Deployment
The trained model can be deployed as:
- **Chatbot** (Streamlit, Flask, or FastAPI)
- **Web API** using FastAPI
- **Telegram Bot** for legal queries





