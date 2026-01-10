from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings #deprecated
# from langchain_huggingface import HuggingFaceEmbeddings  # need to install langchain-huggingface
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


import os
import joblib





# Extract text from PDF
def load_pdf_files(path):
    loader = DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def extract_data(cachePath="data/extracted_data.joblib", pdfPath="data"): 
    CACHE_PATH = cachePath
    extracted_data = None

    if os.path.exists(CACHE_PATH):
        extracted_data = joblib.load(CACHE_PATH)
        print("Loaded from cache")
    else:
        extracted_data = load_pdf_files(pdfPath)
        joblib.dump(extracted_data, CACHE_PATH)
        print("Extracted and cached")
    
    return extracted_data


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of Docment objects
    containing only the 'source' in metadata and the original page_content
    """

    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        page_num = doc.metadata.get("page")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src, "page": page_num}
            )
        )
    
    return minimal_docs


# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks


def download_embeddings():
    """
    Download and return the HuggingFace embeddings model
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

def create_new_index(index_name:str, pc):
    if not pc.has_index(index_name):
        pc.create_index(
            name = index_name,
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(index_name)
    return index

def upload_vectors(index, index_name, texts_chunk:List[Document], embedding:HuggingFaceEmbeddings):
    
    stats = index.describe_index_stats()
    docsearch = None

    if stats['total_vector_count'] == 0:
        # First time — upload vectors
        docsearch = PineconeVectorStore.from_documents(
            documents=texts_chunk,
            embedding=embedding,
            index_name=index_name
        )
        print("Vectors uploaded")
    else:
        # Already populated — just connect
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding
        )
        print(f"Connected to existing index {index_name} with ({stats['total_vector_count']} vectors)")

    return docsearch

def count_vectors(index):
    stats = index.describe_index_stats()
    return stats['total_vector_count']

def add_new_data(docsearch, content, source, page_no):
    new_data = Document(
        page_content=content,
        metadata={"source":source, "page": page_no}
    )
    docsearch.add_documents(documents=[new_data])

def use_existing_index(index_name, embedding):
    docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding
        )
    return docsearch