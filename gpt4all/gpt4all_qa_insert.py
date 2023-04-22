#Demo python script for using GPT4All with langchain

#This script will read a text document
#Split the document into chunks of desired size
#Create vectorstore of text & embeddings
#Perisitently store vectorstore in Chromadb

#Chroma is a database for building AI applications with embeddings
#Chromadb URL: https://docs.trychroma.com/getting-started

#GPT4All is an open-source assistant-style large language model that can be installed and run locally from a compatible machine. 
#The AI model was trained on 800k GPT-3.5-Turbo Generations based on LLaMa, and can give results similar to OpenAI’s GPT3 and GPT3.5.
#The standard install of GPT4All will use CPU for processing power
#GPT4All URL: https://github.com/nomic-ai/gpt4all

#LangChain is a framework for developing applications powered by language models
#Langchain URL: https://python.langchain.com/en/latest/index.html



# GPT4All model path
GPT4ALL_MODEL_PATH = "/home/saurabh/models/gpt4all-lora-quantized-ggml.bin"
# DB persistent directory path (using the same name as text file name)
DB_DIR ="/home/saurabh/db/state_of_the_union"
# Collection name (using the same name as text file name)
DB_C_Name = "state_of_the_union"

# Loading document to implement Q&A
print("Loading documents...")
from langchain.document_loaders import TextLoader
loader = TextLoader('/home/saurabh/data/state_of_the_union.txt')

# Splitting the document into chunks of desired size
from langchain.text_splitter import CharacterTextSplitter
documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# The embedding layer converts each word in the input text into a high-dimensional vector representation. 
# These embeddings capture semantic and syntactic information about the words and help the model to understand the context.
print("Use llama.cpp embedding model...")
from langchain.embeddings import LlamaCppEmbeddings
embeddings = LlamaCppEmbeddings(model_path=GPT4ALL_MODEL_PATH)


# Create your index directly from loader
print("Creating index...")
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator(embedding=embeddings,
                                vectorstore_kwargs={"persist_directory": DB_DIR, "collection_name":DB_C_Name}
                               ).from_loaders([loader])

# # Storing text collection & embedding details in Chroma db in a persistent manner.
# print("Storing text & embedding details in Chroma db...")
# from langchain.vectorstores import Chroma
# db = Chroma.from_documents(texts, embeddings,collection_name = DB_C_Name, persist_directory= DB_DIR)

print("DB persistence Completed..")



