#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Простой тестовый раннер для RAG-цепочки.
- По желанию пересобирает/обновляет индекс из data_dir
- Вызывает RAG (LM Studio + FAISS) и печатает ответ + JSON-фреймы

Запуск (минимум):
  export OPENAI_BASE_URL="http://127.0.0.1:1234/v1"
  export OPENAI_API_KEY="lm-studio"      # любое непустое значение
  export OPENAI_MODEL="mistral-7b-instruct-v0.2"

  python test_rag.py \
    --data-dir ./data \
    --index-dir ./data/index \
    --rebuild \
    --query "Давай поиграем, где нужно угадывать эмоции" \
    --age 8-10 \
    --filters '{"topic":"emotions","age_band":"8-10"}'

Если индекс уже собран, можно запускать без --rebuild.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Импорт сборки индекса и вызова RAG
try:
    from build_or_update_index import ingest
except Exception as e:
    ingest = None  # обработаем ниже

try:
    from rag_chain import invoke_rag
except Exception as e:
    print("[ERR] Не найден rag_chain.py рядом. Убедитесь, что файл существует.")
    raise


def main():
    ap = argparse.ArgumentParser(description="Тест RAG (LM Studio + LangChain + FAISS)")
    ap.add_argument("--data-dir", type=str, default="./data", help="Каталог с документами (.md/.yaml)")
    ap.add_argument("--index-dir", type=str, default="./data/index", help="Каталог индекса FAISS")
    ap.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Модель эмбеддингов")

    ap.add_argument("--rebuild", action="store_true", help="Пересобрать индекс с нуля перед тестом")

    ap.add_argument("--query", type=str, required=True, help="Пользовательский запрос/реплика")
    ap.add_argument("--age", type=str, default="8-10", help="Возрастной диапазон (например, 8-10)")
    ap.add_argument("--filters", type=str, help='JSON-строка фильтров RAG, напр. {"topic":"emotions","age_band":"8-10"}')

    ap.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"))
    ap.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", "lm-studio"))
    ap.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "mistral-7b-instruct-v0.2"))
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=600)

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1) Сборка/обновление индекса при необходимости
    if args.rebuild or not (index_dir.exists() and any(index_dir.iterdir())):
        if ingest is None:
            print("[ERR] Не удалось импортировать ingest() из build_or_update_index.py. Положите файл рядом.")
            sys.exit(2)
        if not data_dir.exists():
            print(f"[ERR] Каталог данных не найден: {data_dir}")
            sys.exit(2)
        mode = "rebuild" if args.rebuild else "update"
        print(f"[INFO] Строим индекс ({mode}) из {data_dir} → {index_dir} …")
        added_chunks, added_docs = ingest(data_dir, index_dir, args.encoder, mode)
        print(f"[OK] Индекс обновлён. Чанков добавлено: {added_chunks}, документов: {added_docs}")
    else:
        print("[INFO] Используем существующий индекс.")

    # 2) Разбор фильтров
    filters = None
    if args.filters:
        try:
            filters = json.loads(args.filters)
        except Exception as e:
            print(f"[WARN] Не смог распарсить --filters как JSON: {e}. Игнорирую фильтры.")
            filters = None

    # 3) Вызов RAG
    print("[INFO] Отправляю запрос в RAG…")
    text, frames = invoke_rag(
        args.query,
        age_band=args.age,
        controls={},
        index_dir=str(index_dir),
        encoder_name=args.encoder,
        filters=filters,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # 4) Вывод результатов
    print("\n=== ОТВЕТ МОДЕЛИ ===\n")
    print(text)
    print("\n=== JSON-ФРЕЙМЫ ===\n")
    print(json.dumps(frames, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
