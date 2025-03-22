
# **NyaaySaarthi: AI-Powered Legal Aid Assistant**

## **Overview**
NyaaySaarthi is an AI-driven legal aid assistant designed to help users understand Indian laws by analyzing their legal queries. The system provides insights into relevant laws, applicable legal provisions, and suggested legal actions. It is built using machine learning and NLP techniques, leveraging legal texts from the **Indian Penal Code (IPC)** and **National Legal Services Authority (NALSA) guidelines**.

## **Problem Statement**
Many citizens, especially in rural areas, lack access to legal support due to high costs and limited awareness. Understanding legal procedures and identifying the correct legal provisions can be challenging for individuals without legal expertise. NyaaySaarthi addresses this issue by providing:
- Easy access to legal information based on user queries.
- Automated identification of relevant Indian laws and sections.
- Guidance on the next legal steps, such as which court to approach.

## **Tech Stack Used**
- **Google Colab** – Cloud-based environment for model training and testing.
- **PyMuPDF (fitz)** – Extracts legal text from PDF documents (IPC & NALSA guidelines).
- **Pandas** – Processes and structures extracted legal text into a usable dataset.
- **Hugging Face Transformers** – Fine-tunes GPT-2/InLegalBERT for legal question answering.
- **PyTorch** – Implements deep learning models for training and inference.
- **Gradio** – Deploys an interactive web-based UI for user queries.

## **Dataset Collection**
- **Indian Penal Code (IPC) PDF** – Contains legal provisions related to criminal laws in India.
- **National Legal Services Authority (NALSA) Guidelines PDF** – Provides legal aid and rights-related information.
- **Text Extraction with PyMuPDF (fitz)** – Converts legal documents into structured text for model training.
- **Data Preprocessing with Pandas** – Cleans and formats legal text for training.
- **Hugging Face Datasets Library** – Structures and tokenizes data for efficient model learning.

## **Project Workflow**

### **User Flow (How Users Interact with the System)**
1. The user enters a legal query in natural language.
2. The AI model processes the query and extracts key legal terms.
3. Relevant legal provisions from the IPC and NALSA dataset are retrieved.
4. The AI generates a response explaining the applicable law and suggesting next steps.
5. The user receives guidance on which court to approach or whether a lawyer is needed.

### **Tech Flow (Backend Processing Mechanism)**
1. **Legal Data Collection** – Extracts structured legal text from official documents.
2. **Model Training** – Fine-tunes GPT-2/InLegalBERT on legal text datasets.
3. **Query Processing** – Tokenizes and processes user input for legal understanding.
4. **Response Generation** – The model generates relevant legal explanations.
5. **Deployment via Gradio** – Provides an interactive web-based interface for user queries.

## **Code Snippets**
### **Extracting Text from Legal PDFs**
```python
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

pdf_text = extract_text_from_pdf("IPC.pdf")
print(pdf_text[:500])  # Print first 500 characters
```

### **Training the Legal NLP Model**
```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments

model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### **Deploying with Gradio**
```python
import gradio as gr

def predict_law(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model(**inputs)
    return "Relevant Legal Provision: Section 420 IPC"

demo = gr.Interface(fn=predict_law, inputs="text", outputs="text")
demo.launch()
```

## **Future Scope**
- **Integration with Judicial Case Laws** – Enhancing accuracy by using past court judgments.
- **Multi-Language Support** – Expanding legal assistance to regional languages.
- **Mobile App & WhatsApp Bot Deployment** – Making the service accessible on multiple platforms.
- **Lawyer Recommendation System** – Connecting users with legal professionals based on their case.
- **Automated Legal Document Drafting** – Helping users generate petitions and affidavits.
- **Integration with Government Legal Aid Services** – Providing official legal assistance channels.
- **Advanced NLP for Contextual Understanding** – Improving accuracy for complex legal queries.

## **How to Run the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/scitech-butterfly/NyaaySaarthi.git
   cd NyaaySaarthi
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Gradio interface:
   ```bash
   python app.py
   ```
4. Open the link generated by Gradio to interact with the legal AI assistant.

## **Contributing**
Contributions are welcome! If you find bugs, have feature requests, or want to improve the model, feel free to create an issue or submit a pull request.

## **License**
This project is open-source and available under the MIT License.

