from .. import config

COLLECTION = "run_memory"


def should_remember(status: str, final_text: str) -> bool:
    """Only real answers are recalled later; refusals and escalations would pollute context."""
    text = (final_text or "").strip()
    return status == "answer" and bool(text) and text != config.REFUSAL_TEXT


class SentenceTransformerEmbedder:
    """Local embeddings (all-MiniLM-L6-v2). The model loads on first use."""

    def __init__(self, name: str = "all-MiniLM-L6-v2"):
        self.name, self._model = name, None

    def __call__(self, texts):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()


class ChromaMemory:
    def __init__(self, collection, embed):
        self.col, self.embed = collection, embed

    @classmethod
    def from_config(cls, embed=None, name: str = COLLECTION):
        import chromadb
        client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
        return cls(client.get_or_create_collection(name), embed or SentenceTransformerEmbedder())

    def remember(self, run_id: int, prompt: str, answer: str, agent: str = "supervisor") -> None:
        self.col.upsert(
            ids=[str(run_id)], embeddings=self.embed([prompt]), documents=[prompt],
            metadatas=[{"run_id": int(run_id), "agent": agent,
                        "prompt": prompt[:1000], "answer": answer[:2000]}])

    def search(self, query: str, k: int = 3, min_similarity: float = 0.0) -> list[dict]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_embeddings=self.embed([query]), n_results=min(k, n),
                             include=["metadatas", "distances"])
        out = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            sim = max(0.0, min(1.0, 1.0 - dist / 2.0))  # unit vectors: squared L2 = 2 - 2*cos
            if sim >= min_similarity:
                out.append({"run_id": int(meta["run_id"]), "prompt": meta["prompt"],
                            "answer": meta["answer"], "similarity": round(sim, 4)})
        return out

    def forget(self, run_id: int) -> None:
        self.col.delete(ids=[str(run_id)])

    def count(self) -> int:
        return self.col.count()
