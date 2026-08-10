import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


def load_sample(path = "./sample")
	# Define the directory path
	dir_path = Path(path)
	docs: list[Document] = []

	for file_path in dir_path.iterdir():
	    if file_path.is_file():
	        content = file_path.read_text(encoding="utf-8")
		source = file_path.name
	        print(f"---Just read {source} ---")
		
		docs.append(Document(page_content=content, metadata={"source": source}))
	
	return docs


docs = load_sample()
print(f"Loaded {len(docs)} samples.")
