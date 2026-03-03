import logging

logger = logging.getLogger(__name__)


def index_chunks(collection, chunks, embeddings):
    """Batch-insert chunks with embeddings into an existing Weaviate collection."""
    logger.info(f"Inserting {len(chunks)} chunks into Weaviate")
    with collection.batch.fixed_size(batch_size=1000) as batch:
        for chunk, embedding in zip(chunks, embeddings):
            properties = {
                "text": chunk.text,
                "document_title": chunk.document_title,
                "source_path": str(chunk.source_path),
                "chunk_index": chunk.chunk_index,
            }
            if chunk.parent_text:
                properties["parent_text"] = chunk.parent_text
            batch.add_object(properties=properties, vector=embedding.tolist())
    logger.info("Insertion complete")
