from sentence_transformers import CrossEncoder

# Lazy-loaded reranker models
_models = {}


def _get_model(model):
    if model not in _models:
        _models[model] = CrossEncoder(model, device="mps")
    return _models[model]


def rerank(query, texts, model="BAAI/bge-reranker-base", top_n=None):
    """Rerank texts by relevance to query using a cross-encoder.

    Returns indices of texts sorted by descending relevance score.
    """
    cross_encoder = _get_model(model)
    pairs = [[query, text] for text in texts]
    scores = cross_encoder.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if top_n is not None:
        ranked_indices = ranked_indices[:top_n]
    return ranked_indices
