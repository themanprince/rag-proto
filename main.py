import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from fastapi import FastAPI
import uvicorn
import os
from dotenv import load_dotenv

# Load the environment variables from the .env fil
load_dotenv()


def load_sample(path = "./sample"):
	docs: list[Document] = []
	dir_path = Path(path)
	if not dir_path.exists():
		return docs


	for file_path in dir_path.iterdir():
		if file_path.is_file():
			content = file_path.read_text(encoding="utf-8")
			source = file_path.name
			print(f"---Just read {source} ---")
		
			docs.append(Document(page_content=content, metadata={"source": source}))
	
	return docs


app = FastAPI()

@app.get("/")
def home():
	docs = load_sample()
	return {"message": f"Loaded {len(docs)} samples."}


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 8000))

	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

