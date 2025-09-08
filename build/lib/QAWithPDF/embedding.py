# embedding.py

import os
import sys

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
# from llama_index.core import Settings  # <- only if you prefer the global Settings approach

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

from exception import customexception
from logger import logging


def download_gemini_embedding(model, document, persist_dir: str = "storage"):
    """
    Build (or load) a VectorStoreIndex using Gemini embeddings and return a QueryEngine.
    - model: your LLM (e.g. llama_index.llms.gemini.Gemini(...))
    - document: list of LlamaIndex Document objects
    """
    try:
        logging.info("Creating Gemini embedding model and text splitter...")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        splitter = SentenceSplitter(chunk_size=800, chunk_overlap=20)

        # If an index already exists on disk, load it; otherwise build and persist
        if os.path.isdir(persist_dir) and os.listdir(persist_dir):
            logging.info(f"Loading existing index from '{persist_dir}'...")
            storage_ctx = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_ctx, embed_model=gemini_embed_model)
        else:
            logging.info("Building index from documents (first run)…")
            index = VectorStoreIndex.from_documents(
                document,
                llm=model,                      # NEW: pass LLM directly
                embed_model=gemini_embed_model, # NEW: pass embedding model directly
                transformations=[splitter],     # NEW: chunking without ServiceContext
            )
            index.storage_context.persist(persist_dir=persist_dir)
            logging.info(f"Index persisted to '{persist_dir}'")

        # Create a query engine; you can tune similarity_top_k, etc.
        query_engine = index.as_query_engine(llm=model, similarity_top_k=3)
        return query_engine

    except Exception as e:
        raise customexception(e, sys)
