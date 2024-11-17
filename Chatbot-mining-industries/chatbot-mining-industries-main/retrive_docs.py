from langchain.vectorstores import Pinecone
from typing import Any


class VectorDB:
    def __init__(self, embeddings: Any, index_name: str):
        self.embeddings = embeddings
        self.index_name = index_name
        self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

    def _run(
            self,
            query: str,
    ) -> str:
        similar_docs = self.docsearch.similarity_search_with_score(query, k=2)
        
        return str(similar_docs)
