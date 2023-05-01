from multiprocessing import cpu_count
from multiprocessing import Process
from langchain.llms import GPT4All
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from filesplit.split import Split
from os import stat
from datetime import datetime

GPT4ALL_MODEL_PATH = "/home/saurabh/models/gpt4all-lora-quantized-ggml.bin"
CPU_COUNT = cpu_count()
SAVED_FILE = "/home/saurabh/data/state_of_the_union"
SPLIT_FILE_DIR = "/home/saurabh/data/splitfiles"

def splitfile():
  file_size = stat(SAVED_FILE) 
  print("Size of file :", file_size.st_size, "bytes") 
 # CHUNK_SIZE = round(file_size.st_size/ CPU_COUNT)
  CHUNK_SIZE = 500
  print ("Chunk Size::", CHUNK_SIZE)
  split = Split(SAVED_FILE, SPLIT_FILE_DIR)
  split.bysize(CHUNK_SIZE)

def gpt4allsummarize(chunk_id):
  CHUNK_FILE = SPLIT_FILE_DIR +"/" + "state_of_the_union_" + chunk_id
  print("Chunk File Name ::", CHUNK_FILE)
  llm=GPT4All(model=GPT4ALL_MODEL_PATH, n_ctx=1500, n_threads=4, temp=0.9)
  text_splitter = CharacterTextSplitter()
  # Split the file into smaller chunks of 1000 bytes each and then summarize each chunk
  CHUNK_SIZE = 1000
  SUMMARY = ''
  with open(CHUNK_FILE) as f:
      doc_to_summarize = f.read()
      while doc_to_summarize:
        texts = text_splitter.split_text(doc_to_summarize)
        docs = [Document(page_content=t) for t in texts[:3]]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        SUMMARY = chain.run(docs)
  print(SUMMARY)

if __name__ == '__main__':  
    splitfile()
    start_time = datetime.now()
    gpt4allsummarize('1')
    end_time = datetime.now()
    total_time = end_time - start_time
    print(total_time)
    # # Create process
    # p1 = Process(target=gpt4allsummarize, args=('1',))
    # # Start Process
    # p1.start()
    # # Join Process
    # p1.join()