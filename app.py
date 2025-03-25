from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore

# ✅ Use HuggingFaceHub instead of OpenAI/Ollama
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.prompt import *

# ==================== APP SETUP ====================

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# Set API key for Hugging Face
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ==================== RAG SETUP ====================

# Get sentence-transformers embeddings
embeddings = download_hugging_face_embeddings()

# Load vector index from Pinecone
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Load LLM from HuggingFace
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.4, "max_length": 1000}
)

# Prompt setup
system_prompt = (
    "You are a knowledgeable medical assistant. "
    "Use the following pieces of retrieved context to answer the user's question with a detailed explanation. "
    "Do not leave out important information. If the answer is not in the context, just say you don't know."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ==================== FLASK ROUTES ====================

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Query:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

# ==================== RUN ====================

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
