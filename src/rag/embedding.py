import logging

import numpy as np
from litellm import embedding as litellm_embedding
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy-loaded local models
_local_models = {}

# Models that should be loaded locally via sentence-transformers
LOCAL_PREFIXES = ("BAAI/", "sentence-transformers/")


def _get_local_model(model):
    if model not in _local_models:
        _local_models[model] = SentenceTransformer(model, device="mps")
    return _local_models[model]


def embed_chunks(texts, model="BAAI/bge-base-en-v1.5"):
    if any(model.startswith(prefix) for prefix in LOCAL_PREFIXES):
        local_model = _get_local_model(model)
        return local_model.encode(texts, batch_size=256, show_progress_bar=True)

    all_embeddings = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(f"Embedding batch {i // batch_size + 1} ({len(batch)} texts)")
        response = litellm_embedding(model=model, input=batch)
        all_embeddings.extend([item["embedding"] for item in response.data])
    return np.array(all_embeddings)
