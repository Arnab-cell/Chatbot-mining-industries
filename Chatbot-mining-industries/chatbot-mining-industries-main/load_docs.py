import pinecone
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


with open('pinecone-key.txt', 'r') as f:      
    api_key = f.read()      

with open('openai_key.txt', 'r') as f:      
    openai_key = f.read()

pinecone.init(      
	api_key=api_key,      
	environment='gcp-starter'      
)      
index = pinecone.Index('mining')

os.environ["OPENAI_API_KEY"] = openai_key

class LoadDocs:
    def __init__(self, directory):
        self.directory = directory
        self.embeddings = OpenAIEmbeddings()

    def load_dir(self, directory):
        loader = DirectoryLoader(directory)
        documents = loader.load()
        return documents

    def split_docs(self,chunk_size=1000,chunk_overlap=20):
        documents = self.load_dir(self.directory)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        return docs

    def docs_load(self):
        index = Pinecone.from_documents(self.split_docs(), self.embeddings, index_name='mining')
        return index
    
load_data = LoadDocs('data')
load_data.docs_load()
print("Documents loaded successfully!")