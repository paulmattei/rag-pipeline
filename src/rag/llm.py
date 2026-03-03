import logging
import os

from dotenv import load_dotenv
from litellm import completion

from rag.config import SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)

HYDE_SYSTEM_PROMPT = """\
You are a technical documentation writer for Weaviate, an open-source vector database.
Given a user question, write a short documentation passage (1-2 paragraphs) that would \
answer the question. Write as if this passage comes directly from the official Weaviate docs. \
Include specific technical details, commands, and version numbers where relevant. \
Do not hedge or say "I don't know" — just write the documentation passage."""


def generate_response(prompt) -> tuple[str, int, int]:
    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    response = completion(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1000,
        temperature=0.0,
     #   top_p=1.0,
        messages=messages,
    )
    input_tokens=response.usage.prompt_tokens
    output_tokens=response.usage.completion_tokens
    return response.choices[0].message.content, input_tokens, output_tokens


def generate_hyde_document(query):
    """Generate a hypothetical document passage that would answer the query."""
    messages = [
        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = completion(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=300,
        temperature=0.0,
        messages=messages,
    )
    hypothetical_document = response.choices[0].message.content
    logger.info(f"HyDE for '{query[:50]}': {hypothetical_document[:100]}...")
    return hypothetical_document
