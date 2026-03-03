from rag.reranker import rerank


def test_rerank_promotes_relevant_text():
    query = "How do I install Weaviate?"
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Weaviate supports deployment with Docker. Run docker-compose up to start.",
        "Python lists can be sorted with the sort method.",
    ]
    indices = rerank(query, texts)
    assert indices[0] == 1


def test_rerank_returns_all_indices():
    indices = rerank("query", ["a", "b", "c"])
    assert sorted(indices) == [0, 1, 2]


def test_rerank_top_n():
    indices = rerank("query", ["a", "b", "c", "d"], top_n=2)
    assert len(indices) == 2


def test_rerank_single_text():
    indices = rerank("anything", ["only one"])
    assert indices == [0]
