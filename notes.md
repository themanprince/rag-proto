* setup langsmith API key for automatic logging
* for embedding model, I'm using VoyageAI API. First 200Million tokens free... wooooooo!!
* if using Qdrant as VectorStore, the url parameter is that of the particular cluster you created to store the embeddings on qdrant cloud!
* make sure ingestion pipeline is run once, separate from server querying
* if planning to use a boolean flag as checker to determine if the pipeline has been run, consider that the time you tried that using sqlite3 file DB, the info was not persisted due to the time itself not being persisted on consecutive runs
* uvicorn reload option must be set to False for a deployed container
* even if using docker, record dependencies having their version numbers affixed e.g. using something like a requirements.txt so we'd have fastapi-2.5 instead of simply fastapi.. This helps make the image really reproducible
* I pay for openai api key too