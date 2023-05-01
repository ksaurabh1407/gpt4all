from multiprocessing import cpu_count
from multiprocessing import Process
from multiprocessing import Manager
from langchain.llms import GPT4All
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from filesplit.split import Split
from os import stat
from datetime import datetime

GPT4ALL_MODEL_PATH = "/Users/calminabusyworld/llm/model/gpt4all-lora-quantized-ggml.bin"
CPU_COUNT = cpu_count()
SAVED_FILE = "/Users/calminabusyworld/llm/data/state_of_the_union.txt"
SPLIT_FILE_DIR = "/Users/calminabusyworld/llm/data/splitfiles"

# Split the large file into smaller chunks usiing filesplit python package
def splitfile():
  file_size = stat(SAVED_FILE) 
  print("Size of file :", file_size.st_size, "bytes") 
  CHUNK_SIZE = 1000
  #CHUNK_SIZE = round(file_size.st_size / CPU_COUNT)
  print ("Chunk Size::", CHUNK_SIZE)
  split = Split(SAVED_FILE, SPLIT_FILE_DIR)
  split.bysize(CHUNK_SIZE)

# Summarize the file chunks
def gpt4allsummarize(chunk_id, return_dict):
  CHUNK_FILE = SPLIT_FILE_DIR +"/" + "state_of_the_union_" + str(chunk_id)+ ".txt"
  print("Chunk File Name ::", CHUNK_FILE)
  llm=GPT4All(model=GPT4ALL_MODEL_PATH, n_ctx=1500, n_threads=10, temp=0.9)
  text_splitter = CharacterTextSplitter()
  # Split the file into smaller chunks of 1000 bytes each and then summarize each chunk
  with open(CHUNK_FILE) as f:
      doc_to_summarize = f.read()
      while doc_to_summarize:
        texts = text_splitter.split_text(doc_to_summarize)
        docs = [Document(page_content=t) for t in texts[:3]]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        concise_summary = str(chain.run(docs)).split("CONCISE SUMMARY: ")
        return_dict[chunk_id] = concise_summary[1]
        print(concise_summary[1])
        return 1

if __name__ == '__main__':  
    splitfile()
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    chunk_id=0
    range_count = 1
    for i in range(range_count):
        chunk_id += 1
        p = Process(target=gpt4allsummarize, args=(chunk_id, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(return_dict.values())