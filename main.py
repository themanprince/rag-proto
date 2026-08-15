import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_qdrant import QdrantVectorStore
from deepagents.backends import StateBackend
from langchain.tools import tool
from qdrant_client import QdrantClient
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
import uvicorn
import os
from dotenv import load_dotenv
import uuid


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
	required_API_keys = ["VOYAGE_API_KEY", "LANGSMITH_API_KEY", "QDRANT_API_KEY", "QDRANT_URL"]
	for key in required_API_keys:
		if not os.environ.get(key):
			add_log("Error! Needed env var not found")
			raise HTTPException(status=500, detail="needed env var not found.")

	
	embeddings = VoyageAIEmbeddings(model="voyage-3")
		
	qdrant_collection_name = "VectorStore1"
	client = QdrantClient(
		url=os.environ["QDRANT_URL"],
		api_key = os.environ["QDRANT_API_KEY"]
	)
	if client.collection_exists(qdrant_collection_name):
		add_log("Pipeline has already been run")
		return QdrantVectorStore(
			embedding=embeddings,
			collection_name=qdrant_collection_name,
			client=client
		) #vector store
	
	add_log("Got here so pipeline has not already been run")
	docs = load_samples()
	if not docs:
		raise HTTPException(status=500, detail="unable to get files for ingesting")
	
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	all_splits = text_splitter.split_documents(docs)
	add_log(f"Split documentation into {len(all_splits)} chunks.")
		
	qdrant = QdrantVectorStore.from_documents(
		all_splits,
		embeddings,
		url=os.environ["QDRANT_URL"],
		api_key = os.environ["QDRANT_API_KEY"],
		prefer_grpc = False,
		collection_name = qdrant_collection_name
	)
	add_log(f"Indexed {len(all_splits)} chunks")
	
	return qdrant #vector store


backend = StateBackend()

@tool(parse_docstring=True)
def search_documentation(query: str) -> str:

	"""Search LangChain documentation and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """
	
	vector_store_ref = check_and_run_ingestion_pipeline()
	retrieved_docs = vector_store_ref.similarity_search(query, k=4)
	batch_id = uuid.uuid4().hex[:8]
	uploads: list[tuple[str, bytes]] = []
	saved_paths: list[str] = []
	
	for index, doc in enumerate(retrieved_docs, start=1):
		path = f"/retrieved/{batch_id}/chunk_{index}.md"
		content = (
			f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
			f"{doc.page_content}"
		)
		uploads.append((path, content.encode("utf-8")))
		saved_paths.append(path)
	
	upload_results = backend.upload_files(uploads)
	failed_paths = [result.path for result in upload_results if result.error]
	if failed_paths:
		raise HTTPException(status_code=500, detail=f"Failed to save: {failed_paths}")
	
	return (
		f"Saved {len(saved_paths)} documentation chunks:\n"
		+ "\n".join(saved_paths)
	)



app = FastAPI()

@app.get("/{query}")
def home(query: str):
	return search_documentation.invoke({"query": query})


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 8000))

	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

