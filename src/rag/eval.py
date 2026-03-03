import logging

from rag.llm_judge import check_retrieval, score_recall

logger = logging.getLogger(__name__)


def compute_expected_match(retrieved_paths, expected_paths):
    """Compare retrieved paths against expected source paths.

    Returns a summary string like '2/5 matched\nResult 1: True\n...'
    """
    match_lines = []
    for i, path in enumerate(retrieved_paths):
        matched = any(path.endswith(exp) for exp in expected_paths)
        match_lines.append(f"Result {i+1}: {matched}")
    matches_found = sum(1 for line in match_lines if "True" in line)
    return f"{matches_found}/{len(retrieved_paths)} matched\n" + "\n".join(match_lines)


def evaluate(query, response, retrieved_objects, expected_sources, key_facts):
    """Score a single query result: source matching, retrieval check, and fact recall.

    Returns a dict with expected_match and optionally recall_score, retrieved, recalled, and notes.
    """
    retrieved_paths = [obj.properties["source_path"] for obj in retrieved_objects]
    result = {
        "expected_match": compute_expected_match(retrieved_paths, expected_sources),
    }

    if key_facts:
        chunks_text = "\n\n---\n\n".join(obj.properties["text"] for obj in retrieved_objects)
        retrieved = check_retrieval(query, key_facts, chunks_text)
        score, recalled, reasoning = score_recall(query, key_facts, response)
        result["recall_score"] = score
        result["retrieved"] = retrieved
        result["recalled"] = recalled
        result["notes"] = reasoning

    return result
