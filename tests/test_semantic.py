import math
import zlib

import pytest

from agentlab import config
from agentlab.memory.semantic import ChromaMemory, should_remember


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().replace("?", " ").replace(".", " ").split():
            v[zlib.crc32(w.encode()) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


class FakeCol:
    """Mimics the Chroma calls we use; distance is squared L2, like Chroma's default space."""

    def __init__(self):
        self.rows = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, e, m in zip(ids, embeddings, metadatas):
            self.rows[i] = (e, m)

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, include):
        q = query_embeddings[0]
        scored = sorted(((sum((a - b) ** 2 for a, b in zip(q, e)), m) for e, m in self.rows.values()),
                        key=lambda x: x[0])[:n_results]
        return {"metadatas": [[m for _, m in scored]], "distances": [[d for d, _ in scored]]}

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)


def mem():
    return ChromaMemory(FakeCol(), fake_embed)


def test_should_remember_rules():
    assert should_remember("answer", "startapp")
    assert not should_remember("answer", config.REFUSAL_TEXT)
    assert not should_remember("answer", "   ")
    assert not should_remember("escalate", "x")
    assert not should_remember("error", "x")


def test_search_ranks_similar_first():
    m = mem()
    m.remember(1, "How do I create a new Django app?", "startapp")
    m.remember(2, "What is 17.5% of 2480?", "434")
    hits = m.search("create a Django app", k=2)
    assert [h["run_id"] for h in hits] == [1, 2] and hits[0]["answer"] == "startapp"


def test_similarity_bounds():
    m = mem()
    m.remember(1, "create a new app", "a")
    top = m.search("create a new app")[0]
    assert top["similarity"] == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= h["similarity"] <= 1.0 for h in m.search("zzz unrelated words"))
    assert m.search("zzz", min_similarity=0.99) == []


def test_search_empty_collection():
    assert mem().search("anything") == []


def test_forget_removes():
    m = mem()
    m.remember(1, "create a new app", "a")
    m.forget(1)
    assert m.count() == 0 and m.search("create a new app") == []


def test_remember_overwrites_same_id():
    m = mem()
    m.remember(1, "create a new app", "old")
    m.remember(1, "create a new app", "new")
    assert m.count() == 1 and m.search("create a new app")[0]["answer"] == "new"


def test_chroma_server_roundtrip():
    chromadb = pytest.importorskip("chromadb")
    name = "agentlab_test_memory"
    try:
        client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
        client.heartbeat()
    except Exception as e:
        pytest.skip(f"Chroma not reachable: {type(e).__name__}")
    try:
        client.delete_collection(name)
    except Exception:
        pass
    m = ChromaMemory(client.get_or_create_collection(name), fake_embed)
    try:
        m.remember(1, "How do I create a new Django app?", "startapp")
        m.remember(2, "What is 17.5% of 2480?", "434")
        hits = m.search("create a Django app", k=2)
        assert hits[0]["run_id"] == 1 and hits[0]["similarity"] > hits[1]["similarity"]
        m.forget(1)
        assert m.count() == 1
    finally:
        client.delete_collection(name)
