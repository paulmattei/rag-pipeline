import logging
import os.path
from collections import defaultdict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def get_credentials():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds


def get_service():
    creds = get_credentials()
    return build("sheets", "v4", credentials=creds)


def read_benchmark(spreadsheet_id):
    """Read the Benchmark tab and return expected sources and answers.

    Returns:
        (expected_sources, benchmark_answers) where:
        - expected_sources: {query: [source_path, ...]}
        - benchmark_answers: {query: answer_text}
    """
    service = get_service()
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range="Benchmark!A2:C")
        .execute()
    )
    rows = result.get("values", [])
    expected_sources = defaultdict(list)
    benchmark_answers = {}
    for row in rows:
        if len(row) < 2:
            continue
        query = row[0]
        urls = row[1].strip().split("\n")
        for url in urls:
            url = url.strip()
            if not url:
                continue
            if "/blob/main/" in url:
                path = url.split("/blob/main/")[1]
            else:
                path = url
            expected_sources[query].append(path)
        if len(row) >= 3:
            benchmark_answers[query] = row[2]
    return dict(expected_sources), benchmark_answers


def format_result_row(strategy, top_k, query, system_prompt, result_objects, expected_paths, response, input_tokens, output_tokens):
    """Format Weaviate results into a spreadsheet row dict."""
    retrieved_titles = []
    retrieved_paths = []
    chunks_returned_lines = []

    for i, obj in enumerate(result_objects):
        props = obj.properties
        retrieved_titles.append(props['document_title'])
        retrieved_paths.append(props['source_path'])
        chunks_returned_lines.append(
            f"--- Result {i+1} ---\n"
            f"Document Title: {props['document_title']}\n"
            f"Source Path: {props['source_path']}\n"
            f"Chunk Index: {props['chunk_index']}\n"
            f"{props['text'][:200]}..."
        )

    return {
        "strategy": strategy,
        "top_k": top_k,
        "query": query,
        "chunks_returned": "\n".join(chunks_returned_lines),
        "prompt": f"System: {system_prompt}\nQuestion: {query}",
        "answer": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retrieved_titles": "\n".join(f"{i+1}. {t}" for i, t in enumerate(retrieved_titles)),
        "retrieved_source_paths": "\n".join(f"{i+1}. {p}" for i, p in enumerate(retrieved_paths)),
        "expected_source_paths": "\n".join(expected_paths),
    }


def create_run_tab(spreadsheet_id, run_name, results):
    """Create a new tab for this run and write results.

    results: list of dicts with keys:
        {strategy, top_k, query, chunks_returned, chunks_relevant,
         prompt, answer, answer_quality, input_tokens, output_tokens, notes}
    """
    service = get_service()

    # Create the new tab
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{"addSheet": {"properties": {"title": run_name}}}]},
    ).execute()

    headers = [
        "Chunking Strategy",
        "top k",
        "Query",
        "Chunks Returned",
        "LLM System + User Prompt",
        "Answer",
        "Recall Score",
        "Input tokens",
        "Output tokens",
        "Notes",
        "Retrieved Document Titles",
        "Retrieved Source Paths",
        "Expected Source Paths",
        "Expected Match",
    ]

    rows = [headers]
    for r in results:
        rows.append([
            r.get("strategy", ""),
            r.get("top_k", ""),
            r.get("query", ""),
            r.get("chunks_returned", ""),
            r.get("prompt", ""),
            r.get("answer", ""),
            r.get("recall_score", ""),
            r.get("input_tokens", ""),
            r.get("output_tokens", ""),
            r.get("notes", ""),
            r.get("retrieved_titles", ""),
            r.get("retrieved_source_paths", ""),
            r.get("expected_source_paths", ""),
            r.get("expected_match", ""),
        ])

    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{run_name}'!A1",
        valueInputOption="RAW",
        body={"values": rows},
    ).execute()
    logger.info(f"Created tab '{run_name}' with {len(results)} result rows")


def _parse_recall_score(score_str):
    """Parse a recall score like '3/5' into (found, expected)."""
    if not score_str or "/" not in score_str:
        return None, None
    parts = score_str.split("/")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None, None


def _short_model_name(model):
    """Shorten a model name for column headers."""
    if model is None:
        return ""
    return model.split("/")[-1]


def update_summary_tab(spreadsheet_id, all_results, key_facts_by_query=None):
    """Create or replace the Summary tab with a grid of results.

    all_results: dict mapping (strategy, embedding_model, retrieval_k, reranker_model) tuples
        to lists of result dicts.
    key_facts_by_query: dict mapping query string to list of fact strings.
    """
    service = get_service()
    tab_name = "Summary"

    # Delete existing Summary tab if present
    sheet_metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sheet in sheet_metadata.get("sheets", []):
        if sheet["properties"]["title"] == tab_name:
            sheet_id = sheet["properties"]["sheetId"]
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"requests": [{"deleteSheet": {"sheetId": sheet_id}}]},
            ).execute()
            break

    # Create fresh Summary tab
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{"addSheet": {"properties": {"title": tab_name}}}]},
    ).execute()

    # Derive dimensions from result keys, preserving insertion order
    embedding_models = list(dict.fromkeys(em for _, em, _, _ in all_results))
    top_k_values = sorted(set(k for _, _, k, _ in all_results))
    reranker_models = list(dict.fromkeys(rm for _, _, _, rm in all_results))
    queries = []
    for results in all_results.values():
        for r in results:
            q = r.get("query", "")
            if q not in queries:
                queries.append(q)

    if key_facts_by_query is None:
        key_facts_by_query = {}

    # Build ordered config columns: (strategy, embedding_model, k, reranker_model)
    config_columns = []
    for strategy in list(dict.fromkeys(s for s, _, _, _ in all_results)):
        for embedding_model in embedding_models:
            for k in top_k_values:
                for reranker_model in reranker_models:
                    config_columns.append((strategy, embedding_model, k, reranker_model))

    # Two sub-columns per config: Retrieved and Answered
    header_config = ["Question", "Fact"]
    header_sub = ["", ""]
    for strategy, embedding_model, k, reranker_model in config_columns:
        embed_name = _short_model_name(embedding_model)
        reranker_name = _short_model_name(reranker_model)
        label = f"{embed_name} k={k}"
        if reranker_name:
            label += f" {reranker_name}"
        header_config.append(label)
        header_config.append("")
        header_sub.append("Retrieved")
        header_sub.append("Answered")

    rows = [header_config, header_sub]
    for query in queries:
        facts = key_facts_by_query.get(query, [])
        for fact_index, fact in enumerate(facts):
            row = [query if fact_index == 0 else "", fact]
            for config_key in config_columns:
                results = all_results.get(config_key, [])
                result = next((r for r in results if r.get("query") == query), None)
                if result:
                    retrieved = result.get("retrieved", [])
                    recalled = result.get("recalled", [])
                    row.append("Y" if fact_index < len(retrieved) and retrieved[fact_index] else "N")
                    row.append("Y" if fact_index < len(recalled) and recalled[fact_index] else "N")
                else:
                    row.append("")
                    row.append("")
            rows.append(row)
        rows.append([])  # blank row between queries

    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{tab_name}'!A1",
        valueInputOption="RAW",
        body={"values": rows},
    ).execute()
    logger.info(f"Updated Summary tab with {len(config_columns)} configurations")
