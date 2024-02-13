from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import pandas as p
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


dir_path = 'data'

# Initialize a list to store loaded documents
all_docs = []

def create_vector_db(dir_path):
    # Iterate over each file in the folder
    for filename in os.listdir(dir_path):
        if filename.endswith(".pdf"):  # Adjust the file extension as needed
            file_path = os.path.join(dir_path, filename)

            # Create a PyPDFLoader object for the current PDF file
            loader = PyPDFLoader(file_path)

            # Load the document
            docs = loader.load()

            # Assuming PyPDFLoader.load() returns a list of documents
            all_docs.extend(docs)

    # Print the total number of loaded documents
    print("Total number of documents loaded:", len(all_docs))


    persist_directory = 'chroma_db_vectorstore'
    embedding = OpenAIEmbeddings(model='text-embedding-ada-002')
    doc_with_embedding = Chroma.from_documents(documents=all_docs, embedding=embedding,persist_directory= persist_directory)
    doc_with_embedding.persist()

    print("vectore store is created at this location: ",persist_directory)


create_vector_db(dir_path)





