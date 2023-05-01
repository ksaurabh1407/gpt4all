from distutils.log import debug
from fileinput import filename
from flask import *  
import random
from datetime import datetime
import chromadb
from chromadb.config import Settings
import multiprocessing as mp

app = Flask(__name__)  

GPT4ALL_MODEL_PATH = "/home/saurabh/models/gpt4all-lora-quantized-ggml.bin"
DB_DIR ="/home/saurabh/db/"
  
@app.route('/')  
def main():  
    return render_template("index.html")  
  
@app.route('/llm', methods = ['GET', 'POST'])  
def success(): 
    if request.method == 'POST':
# Read user inputs
        model = request.form.get("model")   
        usecase =  request.form.get("usecase") 
        query =   request.form.get("query")
        return_msg = str(model) + "::" + str(usecase) + "::" + str (query)

# Read file content from the the form body         
        for filename, file in request.files.items():
# Check the file type         

# Generate a random name to save the file         
           savedfile = file.filename + "_" + str(random.randint(0,9)) + "_" + str(datetime.now().microsecond)
#  Save the file and close the file      
           file.save(savedfile)
           file.close()

# Check if Model selected is GPT4All && Usecase is query document
           if ((model == '1') and (usecase == '1')):
                
                gpt4allinsert(savedfile)
                return_msg = gpt4allretrieve(savedfile,query)

           elif((model == '1') and (usecase == '2')):
                
               return_msg = gpt4allsummarize(savedfile)    
# GPT4All insert the file in Chroma DB


# GPT4All retrieve the response


# Return response
        
    return return_msg

        
def gpt4allinsert(savedfile):
  # Loading document to implement Q&A
  print("Loading documents...")
  from langchain.document_loaders import TextLoader
  loader = TextLoader(savedfile)
  # The embedding layer converts each word in the input text into a high-dimensional vector representation. 
  # These embeddings capture semantic and syntactic information about the words and help the model to understand the context.
  print("Use llama.cpp embedding model...")
  from langchain.embeddings import LlamaCppEmbeddings
  embeddings = LlamaCppEmbeddings(model_path=GPT4ALL_MODEL_PATH, n_batch=10)
  # Create your index directly from loader
  print("Creating index...")
  from langchain.indexes import VectorstoreIndexCreator
  index = VectorstoreIndexCreator(embedding=embeddings,
                                vectorstore_kwargs={"persist_directory": DB_DIR, "collection_name":savedfile}
                               ).from_loaders([loader])
  print("DB persistence Completed..") 
  
def gpt4allretrieve(savedfile, query):
    # ------------------Print the list of collections in ChromaDB------------#
  from chromadb.config import Settings
  client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory= DB_DIR
                                ))
  print(client.list_collections())

# -------------Print the count records in your desired collection-------#
  col_name = savedfile
  
  # print(col_name.count())
  
  # ------------Retrieving Vector Data ---------------#
  from langchain.embeddings import LlamaCppEmbeddings
  from langchain.vectorstores import Chroma
# Creating embedding function for mmr search 
  embeddings = LlamaCppEmbeddings(model_path=GPT4ALL_MODEL_PATH)
# Creating vector db with embedding for mmr search 
  vectordb = Chroma(persist_directory= DB_DIR, collection_name= col_name,embedding_function= embeddings)
  print(" Retrieved Vector Data.....")

# ---- Publishing Results-----------------------#
  from langchain.chains import RetrievalQA
  from langchain.llms import GPT4All
  retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k":2})
  # Retrieve with source document
  # qa = RetrievalQA.from_chain_type(llm=GPT4All(model=GPT4ALL_MODEL_PATH, n_ctx=1024, n_threads=8), chain_type="stuff", retriever=retriever, return_source_documents=True)
  # Retrieve without source document
  qa = RetrievalQA.from_chain_type(llm=GPT4All(model=GPT4ALL_MODEL_PATH, n_ctx=1024, n_threads=6, n_batch=2, temp=0.9), chain_type="stuff", retriever=retriever, return_source_documents=False)
  result = qa({"query": query})
  return result

def gpt4allsummarize(savedfile):
    from langchain.llms import GPT4All
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains.mapreduce import MapReduceChain
    from langchain.prompts import PromptTemplate
    from langchain.docstore.document import Document
    from langchain.chains.summarize import load_summarize_chain
    llm=GPT4All(model=GPT4ALL_MODEL_PATH, n_ctx=3500, n_threads=6, temp=0.5)
    text_splitter = CharacterTextSplitter()
    # Split the file into smaller chunks of 500 bytes each and then summarize each chunk
    CHUNK_SIZE = 1000
    SUMMARY = ''
    with open(savedfile) as f:
      doc_to_summarize = f.read(CHUNK_SIZE)
      while doc_to_summarize:
        texts = text_splitter.split_text(doc_to_summarize)
        docs = [Document(page_content=t) for t in texts[:3]]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        SUMMARY = chain.run(docs)
        print(SUMMARY)
    return str(SUMMARY)
        

if __name__ == '__main__':  
    app.run(debug=True)