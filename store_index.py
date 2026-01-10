from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, extract_data, filter_to_minimal_docs, text_split, download_embeddings, create_new_index, upload_vectors, count_vectors, add_new_data
from pinecone import Pinecone



load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


extracted_data = extract_data(cachePath="data/extracted_data.joblib")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)
embedding = download_embeddings()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name="medical-chatbot"
index = create_new_index(index_name=index_name, pc=pc)

docsearch = upload_vectors(index=index, index_name=index_name, texts_chunk=texts_chunk, embedding=embedding)


# Checks the working of the above code
vectors_count = count_vectors(index=index)
print(f"Total vectors in the index: {vectors_count}")
print("******")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retrieved_docs = retriever.invoke("what is Acne?")
print(retrieved_docs)
