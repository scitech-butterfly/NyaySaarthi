# imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import torch
from transformers import BitsAndBytesConfig
from langchain_community.document_loaders import PyPDFDirectoryLoader
import textwrap
from langchain.document_transformers import LongContextReorder
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser  
from langchain.document_loaders import PyPDFLoader 

     
# Model
name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(name,cache_dir="./model/")
model = AutoModelForCausalLM.from_pretrained(
    name,
    cache_dir="./model/",
    device_map="auto",
    torch_dtype=torch.float16
)
