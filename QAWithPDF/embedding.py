# embedding.py

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging


def download_gemini_embedding(model, document, persist_dir: str = "storage"):
    """
    Build (or load) a VectorStoreIndex using Gemini embeddings and return a QueryEngine.
    Works with LlamaIndex 0.10+ (ServiceContext removed).
    """
    try:
        logging.info("Creating Gemini embedding model and text splitter...")
        embed_model = GeminiEmbedding(model_name="models/embedding-001")
        splitter = SentenceSplitter(chunk_size=800, chunk_overlap=20)

        # Build a fresh index; persist it so subsequent runs can load it quickly.
        logging.info("Building index from documents…")
        index = VectorStoreIndex.from_documents(
            document,
            llm=model,                 # pass LLM directly (no ServiceContext)
            embed_model=embed_model,   # pass embedding model directly
            transformations=[splitter] # chunking without ServiceContext
        )
        index.storage_context.persist(persist_dir=persist_dir)

        logging.info("Creating query engine…")
        return index.as_query_engine(llm=model)

    except Exception as e:
        raise customexception(e, sys)
