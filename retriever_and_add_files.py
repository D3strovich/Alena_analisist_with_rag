#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss

# InMemoryDocstore для пустого индекса
try:
    from langchain_community.docstore import InMemoryDocstore
except Exception:
    from langchain.docstore import InMemoryDocstore  # старые версии

# Embeddings: новый пакет (убирает warning), с фолбэком на старый
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


def _load_or_create_vs(index_dir: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Грузим сохранённый FAISS или создаём пустой корректно (без from_documents([]))."""
    p = Path(index_dir)
    p.mkdir(parents=True, exist_ok=True)

    if (p / "index.faiss").exists() or (p / "faiss.index").exists() or (p / "index.pkl").exists():
        return FAISS.load_local(str(p), embeddings, allow_dangerous_deserialization=True)

    dim = len(embeddings.embed_query("bootstrap"))
    index = faiss.IndexFlatIP(dim)  # dot-product + нормализация
    vs = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )
    return vs


class KidsRAGRetriever:
    """
    Простой ретривер поверх FAISS:
    - базовый поиск + отсечение слабых результатов (MIN_SCORE)
    - лёгкая переранжировка по косинусу к «якорю»
    """

    def __init__(self, index_dir: str, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.encoder_name = encoder_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.encoder_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vs: FAISS = _load_or_create_vs(index_dir, self.embeddings)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        k_search: int = 6,
        filters: Dict[str, Any] | None = None,
    ) -> List[Document]:
        docs_scores: List[Tuple[Document, float]] = self.vs.similarity_search_with_relevance_scores(query, k=k_search)

        # порог релевантности
        MIN_SCORE = 0.30
        docs_scores = [(d, s) for d, s in docs_scores if s >= MIN_SCORE]
        if not docs_scores:
            return []

        # переранжировка близостью к первому (якорному) документу
        if len(docs_scores) > 1:
            anchor_text = docs_scores[0][0].page_content[:512]
            anchor_e = self.embeddings.embed_query(anchor_text)

            def cos(a, b):
                a = np.asarray(a); b = np.asarray(b)
                return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

            rescored: List[Tuple[Document, float]] = []
            for d, _ in docs_scores:
                e = self.embeddings.embed_query(d.page_content[:512])
                rescored.append((d, cos(anchor_e, e)))
            docs_scores = sorted(rescored, key=lambda x: x[1], reverse=True)

        return [d for d, _ in docs_scores[:k]]
