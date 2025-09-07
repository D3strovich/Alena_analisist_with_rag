#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт добавления/обновления данных в RAG-индексе (FAISS) для проекта на LangChain.

Форматы входных файлов:
 1) Markdown (.md) с YAML front matter между тремя дефисами в начале файла:
    ---
    id: psychoedu-breathing-4-4-4
    type: doc            # doc | questions | guide
    topic: emotions      # emotions | school | peers | family | interests | habits | stressors
    audience: kid        # kid | parent
    age_min: 7
    age_max: 12
    tone: kid
    tags: [breathing, anxiety]
    lang: ru
    source: internal
    ---
    **Упражнение “Дыхание 4–4–4”…**

 2) YAML (.yaml/.yml) для игр со структурированными полями.

Что делает скрипт:
 - Рекурсивно обходит директории с данными.
 - Парсит метаданные и контент.
 - Чанкует текст (бережно к спискам).
 - Строит/обновляет FAISS-индекс с эмбеддингами HuggingFace/SBERT.
 - Избегает дублей по content_hash.

Запуск (пример):
  python build_or_update_index.py \
     --data-dir ./data \
     --index-dir ./data/index \
     --encoder sentence-transformers/all-MiniLM-L6-v2 \
     --mode update

Требования:
  - langchain, langchain-community, sentence-transformers, faiss-cpu, pyyaml, numpy
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np

# NEW: добавили эти импорты НИЖЕ future-импорта
import faiss
try:
    from langchain_community.docstore import InMemoryDocstore
except Exception:
    from langchain.docstore import InMemoryDocstore  # на случай другой версии

# Text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ------------------------------
# Утилиты
# ------------------------------

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class ParsedDoc:
    content: str
    metadata: Dict


def parse_markdown_with_front_matter(path: Path) -> ParsedDoc:
    raw = read_text(path)
    m = FRONT_MATTER_RE.match(raw)
    meta: Dict = {}
    body = raw
    if m:
        fm = m.group(1)
        body = m.group(2)
        try:
            meta = yaml.safe_load(fm) or {}
        except Exception:
            meta = {}
    # Мини-очистка
    body = body.strip()
    meta.setdefault("id", str(path.stem))
    meta.setdefault("type", "doc")
    meta.setdefault("source_path", str(path))
    meta.setdefault("ext", path.suffix)
    return ParsedDoc(content=body, metadata=meta)


def parse_game_yaml(path: Path) -> ParsedDoc:
    data = yaml.safe_load(read_text(path)) or {}
    meta = {
        "id": data.get("id", path.stem),
        "type": data.get("type", "game"),
        "topic": data.get("topic"),
        "audience": data.get("audience", "kid"),
        "age_min": data.get("age_min"),
        "age_max": data.get("age_max"),
        "tone": data.get("tone", "kid"),
        "tags": data.get("tags"),
        "title": data.get("title"),
        "goal": data.get("goal"),
        "duration": data.get("duration"),
        "source_path": str(path),
        "ext": path.suffix,
    }
    # Превратим структуру игры в текст для эмбеддинга (сохраняем смысл)
    lines: List[str] = []
    if data.get("title"):
        lines.append(f"Игра: {data['title']}")
    if data.get("goal"):
        lines.append(f"Цель: {data['goal']}")
    if data.get("steps"):
        lines.append("Шаги:")
        for i, s in enumerate(data["steps"], 1):
            lines.append(f"{i}. {s}")
    if data.get("reflection"):
        lines.append("Рефлексия:")
        for i, r in enumerate(data["reflection"], 1):
            lines.append(f"- {r}")
    content = "\n".join(lines).strip()
    return ParsedDoc(content=content, metadata=meta)


SUPPORTED_MD = {".md", ".markdown"}
SUPPORTED_YAML = {".yaml", ".yml"}


def walk_data_dir(data_dir: Path) -> List[ParsedDoc]:
    docs: List[ParsedDoc] = []
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in SUPPORTED_MD:
            parsed = parse_markdown_with_front_matter(p)
            docs.append(parsed)
        elif p.suffix.lower() in SUPPORTED_YAML:
            parsed = parse_game_yaml(p)
            docs.append(parsed)
    return docs


# ------------------------------
# Чанкинг
# ------------------------------

def chunk_parsed_doc(parsed: ParsedDoc, chunk_size_chars: int = 1600, chunk_overlap: int = 200) -> List[Document]:
    """Режем по символам (прибл. 400–900 токенов), бережно к спискам."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n- ", "\n* ", "\n", ". ", " "]
    )
    chunks = splitter.split_text(parsed.content)

    base_meta = {k: v for k, v in parsed.metadata.items()}
    chash = sha1(parsed.content)
    base_meta["content_hash"] = chash

    docs: List[Document] = []
    for i, ch in enumerate(chunks):
        md = dict(base_meta)
        md["chunk_index"] = i
        md["chunk_id"] = f"{base_meta.get('id')}::chunk_{i}"
        docs.append(Document(page_content=ch, metadata=md))
    return docs


# ------------------------------
# Индекс: создание/обновление
# ------------------------------

def ensure_index_dir(index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)


def load_or_create_vectorstore(index_dir: Path, embeddings: HuggingFaceEmbeddings) -> FAISS:
    # если индекс уже есть — пробуем загрузить
    if (index_dir / "index.faiss").exists() or (index_dir / "faiss.index").exists() or (index_dir / "index.pkl").exists():
        try:
            vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            print(f"[INFO] Загружен существующий индекс из {index_dir}")
            return vs
        except Exception as e:
            print(f"[WARN] Не удалось загрузить индекс ({e}). Будет создан новый.")

    # создаём ПУСТОЙ индекс корректно (не вызывая from_documents([]))
    dim = len(embeddings.embed_query("bootstrap"))
    index = faiss.IndexFlatIP(dim)  # косинус через IP (мы нормализуем эмбеддинги)
    vs = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )
    return vs




def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


def filter_new_chunks(chunks: List[Document], existing: FAISS) -> List[Document]:
    """Фильтруем чанки, которые уже есть в индексе по content_hash+chunk_index.
       (простой способ: ищем точные совпадения по этим метаданным в docstore)"""
    try:
        docstore = existing.docstore._dict  # type: ignore[attr-defined]
    except Exception:
        # нет docstore — добавим всё как есть
        return chunks

    existing_keys = set()
    for _id, doc in docstore.items():
        md = getattr(doc, "metadata", {}) or {}
        key = (md.get("content_hash"), md.get("chunk_index"), md.get("id"))
        existing_keys.add(key)

    fresh: List[Document] = []
    for d in chunks:
        md = d.metadata or {}
        key = (md.get("content_hash"), md.get("chunk_index"), md.get("id"))
        if key in existing_keys:
            continue
        fresh.append(d)
    return fresh


def ingest(data_dir: Path, index_dir: Path, encoder: str, mode: str = "update") -> Tuple[int, int]:
    ensure_index_dir(index_dir)
    embeds = build_embeddings(encoder)

    if mode == "rebuild" and index_dir.exists():
        # Очистка каталога индекса
        for f in index_dir.glob("*"):
            try:
                f.unlink()
            except IsADirectoryError:
                pass
        print(f"[INFO] Режим rebuild: каталог {index_dir} очищен")

    vectorstore = load_or_create_vectorstore(index_dir, embeds)

    parsed_docs = walk_data_dir(data_dir)
    all_chunks: List[Document] = []
    for pd in parsed_docs:
        chunks = chunk_parsed_doc(pd)
        all_chunks.extend(chunks)

    if mode == "update":
        all_chunks = filter_new_chunks(all_chunks, vectorstore)

    if not all_chunks:
        print("[INFO] Нет новых документов для добавления.")
        return 0, 0

    print(f"[INFO] Добавляем {len(all_chunks)} чанков в индекс…")
    vectorstore.add_documents(all_chunks)
    vectorstore.save_local(str(index_dir))

    # Простая статистика: кол-во уникальных исходных документов
    unique_sources = len({d.metadata.get("id") for d in all_chunks})
    print(f"[OK] Сохранено. Новых чанков: {len(all_chunks)} | Документов: {unique_sources}")
    return len(all_chunks), unique_sources


# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Добавить файлы в RAG-индекс (FAISS)")
    ap.add_argument("--data-dir", type=str, required=True, help="Каталог с данными (.md/.yaml)")
    ap.add_argument("--index-dir", type=str, required=True, help="Куда класть/где лежит индекс FAISS")
    ap.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Модель эмбеддингов")
    ap.add_argument("--mode", type=str, choices=["update", "rebuild"], default="update", help="update: доиндексировать; rebuild: пересобрать с нуля")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)

    if not data_dir.exists():
        print(f"[ERR] Каталог данных не найден: {data_dir}")
        sys.exit(2)

    added_chunks, added_docs = ingest(data_dir, index_dir, args.encoder, args.mode)
    print(json.dumps({"added_chunks": added_chunks, "added_docs": added_docs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
