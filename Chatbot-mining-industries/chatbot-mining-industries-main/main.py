import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI

# Load API keys
with open('openai_key.txt', 'r') as f:
    openai_key = f.read().strip()
os.environ["OPENAI_API_KEY"] = openai_key

with open('pinecone-key.txt', 'r') as f:
    api_key = f.read().strip()

# Initialize Pinecone
pinecone.init(api_key=api_key, environment='gcp-starter')  # Update with correct environment

# Ensure the Pinecone index exists
index_name = 'mining_regulations'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536)  # Dimension depends on the embedding model

# Initialize the Pinecone vector store
index = pinecone.Index(index_name)
vector_store = Pinecone(index, OpenAIEmbeddings())

# Define the Chatbot prompt template
CHAT_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are an expert on mining laws, acts, rules, and regulations. 
                Given the following context:
                {context}
                Answer the question:
                {question}"""
)

# Initialize the LLM (Language Model)
llm = ChatOpenAI()

# Create a function for the chatbot to handle queries
def mining_chatbot(query: str):
    try:
        # Step 1: Search for relevant context in Pinecone
        context = vector_store.similarity_search(query, k=3)  # You can adjust k for more results
        
        # Step 2: Create the LLMChain to generate responses based on the context
        chain = LLMChain(llm=llm, prompt=CHAT_PROMPT, verbose=True)
        
        # Step 3: Generate response from the chain
        response = chain.run({"question": query, "context": "\n".join([doc.page_content for doc in context])})
        
        return response
    
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage:
question = "What is the procedure for obtaining a mining lease under the Mines and Minerals Act?"
response = mining_chatbot(question)
print(response)
