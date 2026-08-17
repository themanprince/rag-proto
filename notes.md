* setup langsmith API key for automatic logging
* for embedding model, I'm using VoyageAI API. First 200Million tokens free... wooooooo!!
* if using Qdrant as VectorStore, the url parameter is that of the particular cluster you created to store the embeddings on qdrant cloud!
* make sure ingestion pipeline is run once, separate from server querying
* if planning to use a boolean flag as checker to determine if the pipeline has been run, consider that the time you tried that using sqlite3 file DB, the info was not persisted due to the time itself not being persisted on consecutive runs
* uvicorn reload option must be set to False for a deployed container
* even if using docker, record dependencies having their version numbers affixed e.g. using something like a requirements.txt so we'd have fastapi-2.5 instead of simply fastapi.. This helps make the image really reproducible
* I paid for openai api key too
* increase the number of retrieved clusters for better results

### SOME OTHER THINGS I RESEARCHED THAT COULD MAKE THE SOFTWARE BETTER
#### Suggestions for Production-level RAG
1. Do hybrid retrieval
2. ""…..We moved to a layout-aware parser (Docling) that exports to Markdown first. If you don't fix parsing, your retrieval is doomed."
OTHER ALTERNATIVES
        * PyMuPDF (fitz) — Best for mixed PDFs (text + images). but if a page is JUST a scanned image, PyMuPDF cannot read text → you need OCR.

        * OCR using Tesseract For scanned PDFs or images:

1. Re-Ranking is cheap insurance:** We fetch top 50 chunks and re-rank them with a Cross-Encoder. It fixes the 'lost in the middle' problem.
2. **Metadata Filtering:** Don't just search everything. We classify docs by type (Invoice, Contract, Email) during ingestion and filter *before* vector search.
3. Semantic splitting/chunking other than simply RecursiveCharacterTextSplitter e.g. for markdown documents, splitting based on headers within the document
4. Fallback to Vision-LLM:** For the absolute worst cases (nested hierarchies without lines), I crop the table as an **image** and send it to a Vision Model (GPT-4o or Llama 3.2 Vision) with the prompt: *'Transcribe this table to Markdown, preserving the nested hierarchy.'*