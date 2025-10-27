from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI

class AvalaiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.avalai.ir/v1",
        )
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts into a list of vectors.
        """
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query into a vector.

        Args:
            text: The query to embed.

        Returns:
            The embedded vector.
        """
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        """
        Embeds a single text into a vector.

        Args:
            text: The text to embed.

        Returns:
            The embedded vector.
        """

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
