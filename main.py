import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from fastapi import FastAPI, HTTPException
import uvicorn
import os
from dotenv import load_dotenv
from .pipline_has_run_check import PipelineHasRunCheck


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


def check_and_run_ingestion_pipeline():
	pipline_has_run_check = PipelineHasRunCheck()
	
	pipeline_has_already_been_run = pipeline_has_run_check.check()
	
	if pipeline_has_already_been_run:
		add_log("Pipeline has already been run")
		return
	
	add_log("Got here so pipeline has not already been run")
	docs = load_samples()
	if not docs:
		raise HTTPException(status=500, detail="unable to get files for ingesting")
	
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	all_splits = text_splitter.split_documents(docs)
	add_log(f"Split documentation into {len(all_splits)} chunks.")
	
	if not os.environ.get("VOYAGE_API_KEY"):
		add_log("Error! Needed env var not found")
		raise HTTPException(status=500, detail="needed env var not found.")
		
	embeddings = VoyageAIEmbeddings(model="voyage-3")
	qdrant = QdrantVectorStore.from_documents(
		all_splits,
		embeddings,
		url="https://beed57ca-fb89-4356-b2a9-1307eb987f46.australia-southeast1-0.gcp.cloud.qdrant.io",
		prefer_grpc = True,
		collection_name = "VectorStore1"
	)
	add_log(f"Indexed {len(all_splits)} chunks")
	
	pipeline_has_run_check.mark_as_run()
	


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

