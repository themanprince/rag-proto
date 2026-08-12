import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from pathlib import Path
from fastapi import FastAPI, HTTPException
import uvicorn
import os
from dotenv import load_dotenv


# Load the environment variables from the .env fil
load_dotenv()

logs_to_return = [] #custom logging e.g. in place of print statements
add_log = logs_to_return.append

def load_samples(path = "./samples"):
	docs: list[Document] = []
	dir_path = Path(path)
	if not dir_path.exists():
		return docs


	for file_path in dir_path.iterdir():
		if file_path.is_file():
			content = file_path.read_text(encoding="utf-8")
			source = file_path.name
		
			docs.append(Document(page_content=content, metadata={"source": source}))
	
	return docs


ingestion_pipeline_has_been_run = False

def check_and_run_ingestion_pipeline():
	if ingestion_pipeline_has_been_run:
		return
	
	docs = load_samples()
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	all_splits = text_splitter.split_documents(docs)
	add_log(f"Split documentation into {len(all_splits)} chunks.")
	
	if not os.environ.get("GOOGLE_API_KEY"):
		add_log("Error! Needed env var not found")
		raise HTTPException(status=500, detail="needed env var not found.")
		
	embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
	vector_store = InMemoryVectorStore(embeddings)
	vector_store.add_documents(documents=all_splits)
	add_log(f"Indexed {len(all_splits)} chunks")
	
	ingestion_pipeline_has_been_run = True
	


app = FastAPI()

@app.get("/")
def home():
	check_and_run_ingestion_pipeline()
	return {
		"logs": logs_to_return
	}


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 8000))

	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

