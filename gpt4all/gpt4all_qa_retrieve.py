#Demo python script for using GPT4All with langchain

#This script will answer your query based on collection & embedding information in a persisted Chroma DB




import chromadb
from chromadb.config import Settings


# GPT4All model path
GPT4ALL_MODEL_PATH = "/home/saurabh/models/gpt4all-lora-quantized-ggml.bin" 

# DB persistent directory path (using the same name as text file name)
DB_DIR ="/home/saurabh/db/state_of_the_union"

# Collection name (using the same name as text file name)
DB_C_Name = "state_of_the_union"

# Query you want to execute on your embedding database
query = "What did the president say about Ketanji Brown Jackson?"
    

# ------------------Print the list of collections in ChromaDB------------#
from chromadb.config import Settings
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory= DB_DIR
                                ))
print(client.list_collections())

# -------------Print the count records in your desired collection-------#
collection = client.get_collection(DB_C_Name)
print(collection.count())

# ------------Retrieving Vector Data ---------------#
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
# Creating embedding function for mmr search 
embeddings = LlamaCppEmbeddings(model_path=GPT4ALL_MODEL_PATH)
# Creating vector db with embedding for mmr search 
vectordb = Chroma(persist_directory= DB_DIR, collection_name= DB_C_Name,embedding_function= embeddings)
print(" Retrieved Vector Data.....")

# ---- Publishing Results-----------------------#
from langchain.chains import RetrievalQA
from langchain.llms import GPT4All
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k":2})
# Retrieve with source document
# qa = RetrievalQA.from_chain_type(llm=GPT4All(model=GPT4ALL_MODEL_PATH, n_ctx=1024, n_threads=8), chain_type="stuff", retriever=retriever, return_source_documents=True)
# Retrieve without source document
qa = RetrievalQA.from_chain_type(llm=GPT4All(model=GPT4ALL_MODEL_PATH, n_ctx=1024, n_threads=6, temp=0.9), chain_type="stuff", retriever=retriever, return_source_documents=False)
result = qa({"query": query})
print(result)


# # ---- 1st way to retrieve -----------------------#
# print("1st way to retrieve")
# retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":2})
# print(retriever.get_relevant_documents(query))


#--------2nd way to retrieve------------------------#
# print("2nd way to retrieve")

# MIN_DOCS = 1
# from langchain.chains import RetrievalQA
# from langchain.llms import GPT4All
# llm = GPT4All(model="../models/gpt4all-lora-quantized-ggml.bin", n_ctx=1024, n_threads=8)
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectordb.as_retriever(search_kwargs={"k": MIN_DOCS}))
# print(query)
# print(qa.run(query))

# #--------3rd way to retrieve------------------------#
# print("3rd way to retrieve")

# from langchain.chains import RetrievalQA
# from langchain.llms import GPT4All
# llm = GPT4All(model="../models/gpt4all-lora-quantized-ggml.bin", n_ctx=1024, n_threads=8)
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectordb.as_retriever()
# print(query)
# print(qa.run(query))