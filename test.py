from datetime import datetime
import random
import chromadb

DB_DIR ="/home/saurabh/db/"
print (datetime.now().microsecond)

savedfile = "filename" + "_" + str(random.randint(0,9)) + "_" + str(datetime.now().microsecond)
print (savedfile)

# ------------------Print the list of collections in ChromaDB------------#
from chromadb.config import Settings
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory= DB_DIR
                                ))
print(client.list_collections())

# 500 B >> 0:01:52.234847