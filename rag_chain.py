#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG-цепочка для детского ассистента-психолога (без игр) с жёстким анти-эхо.
- LM Studio (OpenAI-compatible) + LangChain + наш FAISS-ретривер
- Короткий ответ на РУССКОМ + РОВНО ОДИН JSON-блок (profile_update | report_event)
- Постобработка: удаление эхо/английского, переписывание «я/мне…» в нейтральное «ты/здорово/понимаю…»
"""
from __future__ import annotations

import os, json, re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document

from retriever_and_add_files import KidsRAGRetriever

# ──────────────────────────────────────────────────────────────────────────────
# ПРОМПТ (игры исключены, говорим от второго лица, без «я/мне…» про ребёнка)
SYSTEM_PROMPT = (
    "Ты — доброжелательный детский психолог. Отвечай ТОЛЬКО НА РУССКОМ, коротко и понятно (1–3 предложения).\n"
    "Обращайся к ребёнку на «ты». НЕ описывай свои предпочтения и чувства (не начинай предложения с «Я…», «Мне…»).\n"
    "Не повторяй дословно слова ребёнка. Не используй английские слова. Не предлагай игры.\n"
    "Если замечаешь риск (самоповреждение/насилие/угрозы) — мягко попроси позвать взрослого.\n\n"
    "В КОНЦЕ ответа выведи РОВНО ОДИН валидный JSON-блок в тройных обратных кавычках ```json ...``` без комментариев.\n"
    "Допустимые типы JSON:\n"
    "- 'profile_update' — можно добавлять: mood_now, interests[], school.likes[], school.difficulties[], "
    "peers.close_friend, family_interaction.evening_time, habits.sleep_hours, habits.screen_time, stressors[], "
    "risk_flags[], evidence.\n"
    "- 'report_event' — поля: theme (emotions|interests|school|peers|family|habits|stressors|risk|other) и summary; "
    "дополнительно confidence (0..1), evidence_refs[].\n"
    "Если ребёнок сообщает о себе — используй 'profile_update'. Если описывает ситуацию — 'report_event'."
)

USER_TEMPLATE = (
    "Возрастная группа: {age_band}\n"
    "Управление: {controls}\n\n"
    "Правила ответа:\n"
    "• 1–3 коротких предложения.\n"
    "• В конце задай 1 понятный вопрос по теме.\n"
    "• Говори от второго лица («ты»), не используй «я/мне» про себя.\n"
    "• Не дублируй слова пользователя.\n\n"
    "[КОНТЕКСТ]\n{context}\n\n"
    "[ПОЛЬЗОВАТЕЛЬ]\n{question}\n\n"
    "[ОТВЕТ] СНАЧАЛА напиши ответ (1–3 предложения) на русском. ПОТОМ, на новой строке, выведи РОВНО ОДИН валидный JSON-блок в ```json ...```."
)

# Пара нейтральных few-shot (JSON экранирован {{…}})
FEWSHOTS = [
    (
        "human",
        "Возрастная группа: 8-10\nУправление: пусто\n[КОНТЕКСТ]\n(интересы)\n"
        "[ПОЛЬЗОВАТЕЛЬ]\nЯ люблю математику и читать книги.\n[ОТВЕТ] СНАЧАЛА текст, потом один JSON."
    ),
    (
        "ai",
        "Здорово, что тебе нравятся задачи и книги. Что больше всего увлекает — решать примеры или читать истории?\n\n"
        "```json\n"
        "{{\"type\":\"profile_update\",\"interests\":[\"математика\",\"чтение\"]}}\n"
        "```"
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# ПАРСЕР JSON (только profile_update|report_event)
JSON_BLOCK_RE = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*\})\s*$", re.IGNORECASE)
INLINE_FRAME_RE = re.compile(r"[\(\[\s](profile_update|report_event)\s*:\s*(\{[\s\S]*?\})", re.IGNORECASE)

def _coerce_json_like(s: str) -> dict | None:
    if not isinstance(s, str): return None
    t = s.strip().replace("'", '"')
    t = re.sub(r'(?P<pre>[{,\s])(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:', r'\g<pre>"\g<key>":', t)
    t = re.sub(r'\bTrue\b', 'true', t); t = re.sub(r'\bFalse\b', 'false', t); t = re.sub(r'\bNone\b', 'null', t)
    try: return json.loads(t)
    except Exception: return None

def extract_frames(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    frames: List[Dict[str, Any]] = []
    clean = text or ""

    matches = list(JSON_BLOCK_RE.finditer(clean))
    if matches:
        last = matches[-2:] if len(matches) >= 2 else matches[-1:]
        for m in last:
            block = m.group(1) or m.group(2)
            try:
                data = json.loads(block)
                cand: List[Dict[str, Any]] = []
                if isinstance(data, dict):
                    if data.get("type") in {"profile_update", "report_event"}:
                        cand.append(data)
                    else:
                        for k, v in list(data.items()):
                            kt = str(k).lower()
                            if kt in {"profile_update", "report_event"} and isinstance(v, dict):
                                v = dict(v); v["type"] = kt; cand.append(v)
                elif isinstance(data, list):
                    for it in data:
                        if isinstance(it, dict) and len(it) == 1:
                            (k, v), = it.items()
                            kt = str(k).lower()
                            if kt in {"profile_update", "report_event"} and isinstance(v, dict):
                                v = dict(v); v["type"] = kt; cand.append(v)
                frames.extend(cand)
            except Exception:
                pass
        clean = clean[:matches[0].start()].rstrip()

    if not frames:
        for t, js in INLINE_FRAME_RE.findall(text or ""):
            obj = _coerce_json_like(js)
            if isinstance(obj, dict):
                obj["type"] = t.lower()
                frames.append(obj)
        if frames:
            clean = INLINE_FRAME_RE.sub("", clean).strip()

    return clean, frames

# ──────────────────────────────────────────────────────────────────────────────
# ПОСТ-ОБРАБОТКА (анти-эхо + «ты»-тон)
RUS_ONLY = re.compile(r"[A-Za-z]{3,}")
GREET_RE = re.compile(r"^\s*(привет|здравствуй|добрый день)\b", re.I)
FIRST_PERSON_START = re.compile(r"^\s*(я|мне|у меня)\b", re.I)

STOPWORDS = {
    "я","мы","ты","он","она","они","это","в","на","и","а","но","или","что","как","когда","где",
    "очень","сильно","просто","ещё","еще","тоже","чуть","немного","иногда","бывает","моё","мое",
    "моя","мои","твой","твои","твоё","твоe","у","из","за","для","про","под","над","по","же","ли"
}

def _keywords_from_user(user: str, n: int = 2) -> str:
    words = re.findall(r"[А-Яа-яA-Za-zёЁ0-9\-]+", user.lower())
    words = [w for w in words if len(w) >= 4 and w not in STOPWORDS]
    uniq = []
    for w in words:
        if w not in uniq:
            uniq.append(w)
    return ", ".join(uniq[:n])

def _rewrite_if_echo(user: str, text: str) -> str:
    """Если ответ слишком похож на реплику ребёнка или начинается с «я/мне…» — переписываем."""
    ut = user.strip().lower()
    tt = text.strip().lower()
    # сходство по подстроке/посимвольной метрике
    ratio = SequenceMatcher(None, ut, tt).ratio()
    starts_fp = bool(FIRST_PERSON_START.match(text))
    if ratio >= 0.55 or starts_fp:
        topic = _keywords_from_user(user)  # «математика», «читать книги», …
        if GREET_RE.match(ut):
            return "Привет! Рад тебя видеть. Как прошёл день?"
        if any(k in ut for k in ["грусть","грустно","печально","тревога","страшно","страх","злюсь","злость"]):
            return "Понимаю, такое бывает. Что именно тебя расстроило и кто обычно помогает, когда неприятно?"
        if topic:
            return f"Здорово, что тебе близко: {topic}. Что в этом особенно нравится?"
        # запасной нейтральный ответ
        return "Понимаю тебя. Расскажи чуть подробнее, что для тебя здесь самое важное?"
    return text

def polish_reply(text: str, user: str) -> str:
    if not text: return text
    t = text.strip()
    # уберём англ. куски и «эхо»-строки
    t = RUS_ONLY.sub("", t)
    # склеим и укоротим
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = " ".join(lines)
    # переписываем эхо / первое лицо
    t = _rewrite_if_echo(user, t)
    # гарантируем вопрос в конце
    if "?" not in t:
        t = t.rstrip(".! ") + " Что думаешь?"
    # короче
    return t[:280]

# ──────────────────────────────────────────────────────────────────────────────
# КОНТЕКСТ + СБОРКА
def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs):
        meta = d.metadata or {}
        parts.append(f"[DOC {i}] (topic={meta.get('topic')}, age_min={meta.get('age_min')}, tone={meta.get('tone')})")
        parts.append(d.page_content.strip())
    return "\n".join(parts)

def build_rag_chain(
    index_dir: str,
    encoder_name: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 360,
):
    retr = KidsRAGRetriever(index_dir=index_dir, encoder_name=encoder_name)

    def _retrieve_fn(inp: Dict[str, Any]) -> List[Document]:
        q = str(inp.get("question", ""))
        return retr.retrieve(q, k=3, k_search=6, filters=inp.get("filters"))

    prompt = ChatPromptTemplate.from_messages(FEWSHOTS + [
        ("human", SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE),
    ])

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_base=base_url,
        openai_api_key=api_key,
        top_p=0.9,
        frequency_penalty=0.5,   # сильнее штрафуем повтор
        presence_penalty=0.3,
    )

    chain = (
        {
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            "age_band": RunnablePassthrough() | RunnableLambda(lambda x: x.get("age_band", "8-10")),
            "controls": RunnablePassthrough() | RunnableLambda(lambda x: x.get("controls", {})),
            "context": RunnablePassthrough() | RunnableLambda(_retrieve_fn) | RunnableLambda(_format_docs),
        } | prompt | llm | StrOutputParser()
    )
    return chain

# ──────────────────────────────────────────────────────────────────────────────
# ВЫЗОВ
def invoke_rag(
    question: str,
    age_band: str = "8-10",
    controls: Dict[str, Any] | None = None,
    index_dir: str = "./data/index",
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 360,
    filters: Dict[str, Any] | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    api_key  = api_key  or os.getenv("OPENAI_API_KEY", "lm-studio")
    model    = model    or os.getenv("OPENAI_MODEL", "mistral-7b-instruct-v0.2")

    chain = build_rag_chain(index_dir, encoder_name, base_url, api_key, model, temperature, max_tokens)
    raw = chain.invoke({"question": question, "age_band": age_band, "controls": controls or {}, "filters": filters or {}})

    text, frames = extract_frames(raw)
    return polish_reply(text, question), frames


if __name__ == "__main__":
    t, fr = invoke_rag("Я люблю математику и загадки.")
    print("=== TEXT ===\n", t)
    print("=== FRAMES ===\n", json.dumps(fr, ensure_ascii=False, indent=2))
