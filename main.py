import json
import logging
import os
import sys
from pathlib import Path

import dotenv
import weaviate

from rag.config import (
    CACHE_DIR,
    EMBEDDING_MODELS,
    RERANKER_MODELS,
    RETRIEVAL_K_VALUES,
    SPREADSHEET_ID,
    STRATEGIES,
)
from rag.llm_judge import extract_key_facts
from rag.load import load_documents
from rag.pipeline import run_label, run_queries, setup_collection
from rag.sheets import read_benchmark, update_summary_tab

logger = logging.getLogger(__name__)

KEY_FACTS_PATH = CACHE_DIR / "key_facts.json"


def load_key_facts(benchmark_answers, rebuild=False):
    """Load key facts from cache, or extract and cache them."""
    if KEY_FACTS_PATH.exists() and not rebuild:
        logger.info(f"Loading cached key facts from {KEY_FACTS_PATH}")
        with open(KEY_FACTS_PATH) as f:
            return json.load(f)

    logger.info("Extracting key facts from benchmark answers...")
    key_facts_by_query = {}
    for query, answer in benchmark_answers.items():
        key_facts_by_query[query] = extract_key_facts(query, answer)

    with open(KEY_FACTS_PATH, "w") as f:
        json.dump(key_facts_by_query, f, indent=2)
    logger.info(f"Cached key facts to {KEY_FACTS_PATH}")

    return key_facts_by_query


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("parsing").setLevel(logging.WARNING)
    dotenv.load_dotenv()
    repo_url = os.environ["REPO_URL"]
    repo_dir = Path(os.environ["REPO_DIR"])

    rebuild = "--rebuild" in sys.argv

    documents = load_documents(repo_url, repo_dir, CACHE_DIR, rebuild=rebuild)
    expected_sources, benchmark_answers = read_benchmark(SPREADSHEET_ID)
    key_facts_by_query = load_key_facts(benchmark_answers, rebuild=rebuild)

    all_results = {}

    with weaviate.connect_to_embedded(
        environment_variables={"CLUSTER_ADVERTISE_ADDR": "127.0.0.1", "LOG_LEVEL": "error"},
    ) as client:
        for strategy in STRATEGIES:
            for embedding_model in EMBEDDING_MODELS:
                try:
                    collection = setup_collection(
                        client, documents, strategy, embedding_model, rebuild=rebuild,
                    )
                except Exception:
                    logger.exception(f"Failed to setup collection for {strategy}/{embedding_model}")
                    continue
                for retrieval_k in RETRIEVAL_K_VALUES:
                    for reranker_model in RERANKER_MODELS:
                        try:
                            results = run_queries(
                                collection, expected_sources, key_facts_by_query,
                                strategy=strategy,
                                embedding_model=embedding_model,
                                retrieval_k=retrieval_k,
                                reranker_model=reranker_model,
                            )
                            key = (strategy, embedding_model, retrieval_k, reranker_model)
                            all_results[key] = results
                        except Exception:
                            logger.exception(
                                f"Failed: {run_label(strategy, embedding_model, retrieval_k, reranker_model)}"
                            )

    if all_results:
        update_summary_tab(SPREADSHEET_ID, all_results, key_facts_by_query)


if __name__ == "__main__":
    main()
