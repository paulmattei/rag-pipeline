import dataclasses
import json
import logging
from datetime import datetime

import numpy as np
import tiktoken
import weaviate
import weaviate.classes as wvc

from rag.chunking import create_chunks, source_category
from rag.config import (
    CACHE_DIR,
    HYBRID_ALPHA,
    QUERIES,
    QUERY_TRANSFORM,
    SEARCH_TYPE,
    SPREADSHEET_ID,
    SYSTEM_PROMPT,
    TOP_K,
)
from rag.embedding import LOCAL_PREFIXES, embed_chunks
from rag.eval import evaluate
from rag.llm import generate_hyde_document, generate_response
from rag.models import Chunk
from rag.reranker import rerank
from rag.retrieval import index_chunks
from rag.sheets import create_run_tab, format_result_row

logger = logging.getLogger(__name__)


def _model_slug(model):
    """Convert a model name to a filesystem-safe slug."""
    return model.split("/")[-1]


def load_chunks(documents, strategy, cache_dir, rebuild=False):
    """Load chunks from cache, or build and cache them."""
    strategy_cache = cache_dir / strategy
    strategy_cache.mkdir(exist_ok=True)
    chunks_path = strategy_cache / "chunks.json"

    if chunks_path.exists() and not rebuild:
        logger.info(f"Loading cached chunks for {strategy}")
        with open(chunks_path) as f:
            return [Chunk(**c) for c in json.load(f)]

    chunks = []
    for doc in documents:
        chunks.extend(create_chunks(doc.body, doc.title, doc.source_path, strategy=strategy))

    with open(chunks_path, "w") as f:
        json.dump([dataclasses.asdict(c) for c in chunks], f, default=str)

    return chunks


def load_embeddings(chunks, strategy, embedding_model, cache_dir, rebuild=False):
    """Load embeddings from cache, or build and cache them."""
    slug = _model_slug(embedding_model)
    embeddings_dir = cache_dir / strategy / slug
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = embeddings_dir / "embeddings.npy"

    if embeddings_path.exists() and not rebuild:
        logger.info(f"Loading cached embeddings for {strategy}/{slug}")
        return np.load(embeddings_path)

    embeddings = embed_chunks([chunk.text for chunk in chunks], model=embedding_model)
    np.save(embeddings_path, embeddings)

    return embeddings


def _build_llm_context(objects):
    """Build LLM context from retrieved objects, using parent text when available.

    Deduplicates parent sections so the LLM sees each parent at most once.
    Falls back to child text when no parent is stored.
    """
    seen_parents = set()
    context_chunks = []
    for obj in objects:
        parent_text = obj.properties.get("parent_text")
        if parent_text:
            if parent_text not in seen_parents:
                seen_parents.add(parent_text)
                context_chunks.append(parent_text)
        else:
            context_chunks.append(obj.properties["text"])
    return "\n\n---\n\n".join(context_chunks)


def process_query(collection, query, expected, key_facts, strategy, embedding_model, retrieval_k, reranker_model=None):
    """Retrieve, generate, and evaluate a single query. Returns result dict."""
    if QUERY_TRANSFORM == "hyde":
        hypothetical_document = generate_hyde_document(query)
        query_embedding = embed_chunks([hypothetical_document], model=embedding_model)
    else:
        query_embedding = embed_chunks([query], model=embedding_model)
    return_properties = ["text", "document_title", "source_path", "chunk_index"]
    if strategy == "markdown-optimized":
        return_properties.append("parent_text")

    if SEARCH_TYPE == "hybrid":
        results = collection.query.hybrid(
            query=query,
            vector=query_embedding[0].tolist(),
            alpha=HYBRID_ALPHA,
            limit=retrieval_k,
            return_properties=return_properties,
        )
    else:
        results = collection.query.near_vector(
            near_vector=query_embedding[0].tolist(),
            limit=retrieval_k,
            return_properties=return_properties,
        )

    if reranker_model is not None:
        # Prepend source category to give reranker discriminative context
        texts = []
        for obj in results.objects:
            category = source_category(obj.properties["source_path"])
            prefix = f"[{category}] " if category else ""
            texts.append(prefix + obj.properties["text"])
        reranked_indices = rerank(query, texts, model=reranker_model, top_n=TOP_K)
        results.objects = [results.objects[i] for i in reranked_indices]

    chunks_text = _build_llm_context(results.objects)
    prompt = f"Question: {query}\n\nChunks:\n{chunks_text}"
    response, input_tokens, output_tokens = generate_response(prompt)

    result = format_result_row(
        strategy, retrieval_k, query, SYSTEM_PROMPT,
        results.objects, expected,
        response, input_tokens, output_tokens,
    )
    scores = evaluate(query, response, results.objects, expected, key_facts)
    result.update(scores)

    return result


def _collection_name(strategy, embedding_model):
    slug = _model_slug(embedding_model)
    return f"Chunks_{strategy}_{slug}".replace("-", "_").replace(".", "_")


def run_label(strategy, embedding_model, retrieval_k, reranker_model):
    """Build a human-readable label for a pipeline run."""
    slug = _model_slug(embedding_model)
    label = f"{strategy}_{slug}_k{retrieval_k}"
    if reranker_model is not None:
        label += f"_{_model_slug(reranker_model)}"
    return label


def _filter_oversized_chunks(chunks, max_tokens=8191):
    """Remove chunks that exceed the API token limit."""
    encoder = tiktoken.get_encoding("cl100k_base")
    filtered = [chunk for chunk in chunks if len(encoder.encode(chunk.text)) <= max_tokens]
    skipped = len(chunks) - len(filtered)
    if skipped:
        logger.warning(f"Skipped {skipped} chunks exceeding {max_tokens} token limit")
    return filtered


def setup_collection(client, documents, strategy, embedding_model, rebuild=False):
    """Load chunks, embed, and index into Weaviate. Done once per (strategy, embedding_model)."""
    chunks = load_chunks(documents, strategy, CACHE_DIR, rebuild=rebuild)
    if not any(embedding_model.startswith(prefix) for prefix in LOCAL_PREFIXES):
        chunks = _filter_oversized_chunks(chunks)
    embeddings = load_embeddings(chunks, strategy, embedding_model, CACHE_DIR, rebuild=rebuild)

    collection_name = _collection_name(strategy, embedding_model)
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
    collection = client.collections.create(
        name=collection_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    )
    index_chunks(collection, chunks, embeddings)
    return collection


def run_queries(collection, expected_sources, key_facts_by_query, strategy, embedding_model, retrieval_k, reranker_model=None):
    """Run all queries for a single configuration.

    Returns list of result dicts, one per query.
    """
    label = run_label(strategy, embedding_model, retrieval_k, reranker_model)
    logger.info(f"\n{'='*60}\n{label}\n{'='*60}")

    run_results = []
    for query in QUERIES:
        try:
            expected = expected_sources.get(query, [])
            key_facts = key_facts_by_query.get(query, [])
            result = process_query(
                collection, query, expected, key_facts,
                strategy, embedding_model, retrieval_k=retrieval_k, reranker_model=reranker_model,
            )
            run_results.append(result)

            logger.info(
                f"  {query[:50]:50s} "
                f"Recall: {result.get('recall_score', 'N/A')}"
            )
        except Exception:
            logger.exception(f"Failed to process query: {query}")

    run_name = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    create_run_tab(SPREADSHEET_ID, run_name, run_results)

    return run_results
