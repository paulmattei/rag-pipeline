import json
import logging
import os

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

logger = logging.getLogger(__name__)


def _parse_json_response(content):
    """Parse JSON from LLM response, stripping markdown code fences if present.

    Handles cases where the LLM outputs multiple code blocks (e.g., self-correcting)
    by trying each block until one parses successfully as the expected JSON.
    """
    content = content.strip()
    if "```" in content:
        # Extract all fenced code blocks and try each one (last first, since
        # the LLM sometimes self-corrects with a second block)
        blocks = []
        in_block = False
        current_block = []
        for line in content.split("\n"):
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip() == "```" and in_block:
                in_block = False
                blocks.append("\n".join(current_block).strip())
                current_block = []
                continue
            if in_block:
                current_block.append(line)
        for block in reversed(blocks):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    return json.loads(content)


def extract_key_facts(query, benchmark_answer):
    """Extract key facts from a benchmark answer. Run once per query at startup."""
    response = completion(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1000,
        temperature=0.0,
        messages=[
            {"role": "system", "content": (
                "Extract the key facts from the benchmark answer. "
                "A key fact is a distinct piece of information that a correct answer should contain. "
                "Respond with ONLY a JSON object: {\"key_facts\": [\"fact1\", \"fact2\", ...]}"
            )},
            {"role": "user", "content": f"Question: {query}\n\nBenchmark Answer:\n{benchmark_answer}"},
        ],
    )

    try:
        result = _parse_json_response(response.choices[0].message.content)
        key_facts = result["key_facts"]
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Failed to extract key facts: {response.choices[0].message.content}")
        key_facts = []

    logger.info(f"Extracted {len(key_facts)} key facts for '{query}'")
    return key_facts


def check_retrieval(query, key_facts, chunks_text):
    """Check which key facts are present in the retrieved chunks."""
    if not key_facts:
        return []

    facts_list = "\n".join(f"- {fact}" for fact in key_facts)
    response = completion(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1000,
        temperature=0.0,
        messages=[
            {"role": "system", "content": (
                "You are checking which key facts from a benchmark are present in retrieved document chunks. "
                "For each fact, determine if the chunks contain that information (even if phrased differently). "
                "Respond with ONLY a JSON object: "
                "{\"present\": [true, false, ...]}"
            )},
            {"role": "user", "content": (
                f"Question: {query}\n\n"
                f"Key Facts:\n{facts_list}\n\n"
                f"Retrieved Chunks:\n{chunks_text}"
            )},
        ],
    )

    try:
        result = _parse_json_response(response.choices[0].message.content)
        present = result["present"]
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Failed to parse retrieval check: {response.choices[0].message.content}")
        present = [False] * len(key_facts)

    num_present = sum(1 for p in present if p)
    logger.info(f"Retrieval check for '{query}': {num_present}/{len(key_facts)} facts in chunks")
    return present


def score_recall(query, key_facts, rag_answer):
    """Score which key facts the RAG answer recalls. Run once per query per pipeline run."""
    if not key_facts:
        return "0/0", [], "no key facts extracted"

    facts_list = "\n".join(f"- {fact}" for fact in key_facts)
    response = completion(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1000,
        temperature=0.0,
        messages=[
            {"role": "system", "content": (
                "You are checking which key facts from a benchmark appear in a RAG answer. "
                "For each fact, determine if the RAG answer contains that information. "
                "Respond with ONLY a JSON object: "
                "{\"recalled\": [true, false, ...], \"reasoning\": \"<brief explanation>\"}"
            )},
            {"role": "user", "content": (
                f"Question: {query}\n\n"
                f"Key Facts:\n{facts_list}\n\n"
                f"RAG Answer:\n{rag_answer}"
            )},
        ],
    )

    try:
        result = _parse_json_response(response.choices[0].message.content)
        recalled = result["recalled"]
        reasoning = result.get("reasoning", "")
        num_recalled = sum(1 for r in recalled if r)
        score = f"{num_recalled}/{len(key_facts)}"
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Failed to parse recall response: {response.choices[0].message.content}")
        score = "0/0"
        recalled = [False] * len(key_facts)
        reasoning = response.choices[0].message.content

    logger.info(f"Recall for '{query}': {score} - {reasoning}")
    return score, recalled, reasoning
