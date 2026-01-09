from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

from src.prompt import *
from src.helper import download_embeddings, use_existing_index

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from typing import List
# from langchain.schema import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings #deprecated
# # from langchain_huggingface import HuggingFaceEmbeddings  3 ned to install langchain-huggingface
# from pinecone import ServerlessSpec


# import os
# import joblib

# from src.helper import load_pdf_files, extract_data, filter_to_minimal_docs, text_split, download_embeddings, create_new_index, upload_vectors, count_vectors, add_new_data
# from pinecone import Pinecone

load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

index_name = "medical-chatbot"
embedding = download_embeddings()
docsearch = use_existing_index(index_name, embedding)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

chatModel = ChatGroq(model="groq/compound")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST", "GET"])
def chat():
    msg = request.form["msg"]
    # print(msg)
    # print(type(msg))
    response = rag_chain.invoke({"input": msg})
    print("User: ", msg)
    print("Model: ", response["answer"])
    return str(response["answer"])





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)