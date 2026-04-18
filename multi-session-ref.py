"""멀티세션 RAG 챗봇 — Streamlit + Supabase(pgvector) + OpenAI."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _REPO_ROOT / ".env"
load_dotenv(_ENV_PATH)

_SECRET_ENV_KEYS = ("OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY")


def _merge_streamlit_secrets_into_environ() -> None:
    """Streamlit Cloud / `.streamlit/secrets.toml` 값을 `os.environ`에 넣어 `getenv`·LangChain·SDK와 호환."""
    try:
        sec = st.secrets
    except Exception:
        return
    if not sec:
        return
    for key in _SECRET_ENV_KEYS:
        if key not in sec:
            continue
        val = sec[key]
        if val is not None and str(val).strip():
            os.environ[key] = str(val).strip()


_merge_streamlit_secrets_into_environ()

LOG_DIR = _REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = LOG_DIR / f"multi_session_rag_{datetime.now():%Y%m%d}.log"

_logger = logging.getLogger("multi_session_rag")
_logger.setLevel(logging.WARNING)
if not _logger.handlers:
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setLevel(logging.WARNING)
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.WARNING)
    _ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(_fh)
    _logger.addHandler(_ch)

for _name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)
    logging.getLogger(_name).propagate = False

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

ANSWER_STYLE = """
답변은 반드시 마크다운 헤딩(# ## ###)을 사용하여 구조화하십시오.
주요 주제는 # (H1), 세부 내용은 ## (H2), 구체적 설명은 ### (H3)로 구분하십시오.
답변은 서술형으로 작성하고 존댓말을 사용하십시오. 완전한 문장으로 서술하십시오.
구분선(---, ===, ___)과 취소선(~~텍스트~~)은 사용하지 마십시오.
참조 번호, 각주, 출처 문구, URL 인용 표기는 넣지 마십시오.
"""


def env_status() -> dict[str, str]:
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
        "SUPABASE_URL": os.getenv("SUPABASE_URL", "").strip(),
        "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY", "").strip(),
    }


def get_supabase() -> Client | None:
    s = env_status()
    if not s["SUPABASE_URL"] or not s["SUPABASE_ANON_KEY"]:
        return None
    return create_client(s["SUPABASE_URL"], s["SUPABASE_ANON_KEY"])


def remove_separators(text: str) -> str:
    if not text:
        return text
    out = re.sub(r"~~[^~]*~~", "", text)
    lines = out.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"[-_=]{3,}", stripped):
            continue
        cleaned.append(line)
    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY가 필요합니다.")
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature, api_key=key)


def get_embeddings() -> OpenAIEmbeddings:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY가 필요합니다.")
    return OpenAIEmbeddings(model=EMBED_MODEL, api_key=key, dimensions=EMBED_DIM)


def _format_memory_for_prompt(memory: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in memory:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            parts.append(f"사용자: {content}")
        elif role == "assistant":
            parts.append(f"어시스턴트: {content}")
    return "\n".join(parts)


def ensure_session_row(sb: Client, session_id: str, title: str = "새 세션") -> None:
    sb.table("chat_sessions").upsert(
        {"id": session_id, "title": title},
        on_conflict="id",
    ).execute()


def save_messages(sb: Client, session_id: str, chat_history: list[dict[str, str]]) -> None:
    sb.table("chat_messages").delete().eq("session_id", session_id).execute()
    rows: list[dict[str, Any]] = []
    for i, m in enumerate(chat_history):
        rows.append(
            {
                "session_id": session_id,
                "role": m.get("role", "user"),
                "content": m.get("content", ""),
                "seq": i,
            }
        )
    if rows:
        sb.table("chat_messages").insert(rows).execute()


def load_session_messages(sb: Client, session_id: str) -> list[dict[str, str]]:
    r = (
        sb.table("chat_messages")
        .select("role,content,seq")
        .eq("session_id", session_id)
        .order("seq", desc=False)
        .execute()
    )
    out: list[dict[str, str]] = []
    for row in r.data or []:
        out.append({"role": row["role"], "content": row["content"]})
    return out


def list_sessions(sb: Client) -> list[dict[str, Any]]:
    r = sb.table("chat_sessions").select("id,title,created_at").order("created_at", desc=True).execute()
    return list(r.data or [])


def generate_short_title(llm: ChatOpenAI, first_q: str, first_a: str) -> str:
    prompt = (
        "다음은 채팅의 첫 질문과 첫 답변입니다. 이 대화를 대표하는 짧은 세션 제목을 "
        "한국어로 40자 이내로 한 줄만 출력하세요. 따옴표나 '제목:' 같은 접두어는 쓰지 마세요.\n\n"
        f"질문: {first_q[:1500]}\n\n답변: {first_a[:1500]}"
    )
    out = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(out, "content", str(out))
    if isinstance(content, list):
        content = "".join(
            getattr(b, "text", str(b)) if not isinstance(b, str) else b for b in content
        )
    title = str(content).strip().splitlines()[0].strip()
    return title[:200] if title else "새 세션"


def generate_followup_questions(llm: ChatOpenAI, user_q: str, answer_excerpt: str) -> str:
    default_block = (
        "\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n"
        "1. 관련 내용을 더 자세히 알려 주실 수 있습니까?\n"
        "2. 다른 각도에서 설명해 주실 수 있습니까?\n"
        "3. 실무에 적용할 때 주의할 점이 있습니까?\n"
    )
    prompt = (
        "다음은 사용자 질문과 답변의 일부입니다. 이어서 물어보면 좋은 질문을 정확히 3개만, "
        "한 줄에 하나씩 번호(1. 2. 3.)로 한국어 존댓말로 작성하세요. 다른 설명은 쓰지 마세요.\n\n"
        f"질문: {user_q}\n\n답변 일부:\n{answer_excerpt[:2000]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        content = getattr(out, "content", str(out))
        if isinstance(content, list):
            content = "".join(
                getattr(b, "text", str(b)) if not isinstance(b, str) else b for b in content
            )
        lines = [ln.strip() for ln in str(content).splitlines() if ln.strip()]
        numbered: list[str] = []
        n = 1
        for ln in lines:
            if re.match(r"^\d+[\.)]\s*", ln):
                numbered.append(ln)
            elif n <= 3:
                numbered.append(f"{n}. {ln}")
                n += 1
            if len(numbered) >= 3:
                break
        while len(numbered) < 3:
            numbered.append(f"{len(numbered) + 1}. (추가 질문을 준비 중입니다)")
        body = "\n".join(numbered[:3])
    except Exception as e:
        _logger.warning("후속 질문 생성 실패: %s", e)
        return default_block

    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{body}\n"


def stream_direct_llm(llm: ChatOpenAI, messages: list[BaseMessage], holder: Any) -> str:
    full = ""
    for chunk in llm.stream(messages):
        piece = getattr(chunk, "content", None)
        if piece is None:
            continue
        if isinstance(piece, list):
            piece = "".join(getattr(p, "text", str(p)) if not isinstance(p, str) else p for p in piece)
        full += piece
        holder.markdown(remove_separators(full))
    return remove_separators(full)


def retrieve_by_rpc(
    sb: Client, session_id: str, query: str, embeddings: OpenAIEmbeddings, k: int = 10
) -> list[Document]:
    q_emb = embeddings.embed_query(query)
    try:
        r = sb.rpc(
            "match_vector_documents",
            {
                "filter_session_id": session_id,
                "query_embedding": q_emb,
                "match_count": k,
            },
        ).execute()
    except Exception as e:
        _logger.warning("RPC 검색 실패: %s", e)
        return []

    docs: list[Document] = []
    for row in r.data or []:
        meta = {
            "file_name": row.get("file_name", ""),
            "id": row.get("id"),
            **(row.get("metadata") or {}),
        }
        docs.append(Document(page_content=row.get("content") or "", metadata=meta))
    return docs


def stream_rag_answer(
    llm: ChatOpenAI,
    sb: Client,
    session_id: str,
    embeddings: OpenAIEmbeddings,
    user_q: str,
    memory: list[dict[str, str]],
    holder: Any,
) -> str:
    docs = retrieve_by_rpc(sb, session_id, user_q, embeddings, k=10)
    context = "\n\n".join(d.page_content for d in docs[:10])
    mem_text = _format_memory_for_prompt(memory)
    system = f"""{ANSWER_STYLE}
아래 참고 문맥과 이전 대화를 바탕으로 사용자 질문에 답하십시오. 참고 문맥에 없는 내용은 추측하지 마십시오.

[이전 대화]
{mem_text}

[참고 문맥]
{context}
"""
    messages: list[BaseMessage] = [
        SystemMessage(content=system),
        HumanMessage(content=user_q),
    ]
    return stream_direct_llm(llm, messages, holder)


def insert_vectors_for_files(
    sb: Client,
    session_id: str,
    splits: list[Document],
    file_name: str,
    embeddings: OpenAIEmbeddings,
    batch_size: int = 10,
) -> None:
    texts = [d.page_content for d in splits]
    for i in range(0, len(texts), batch_size):
        batch_docs = splits[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        embs = embeddings.embed_documents(batch_texts)
        rows = []
        for doc, emb in zip(batch_docs, embs, strict=True):
            meta = dict(doc.metadata) if doc.metadata else {}
            meta.setdefault("source", file_name)
            rows.append(
                {
                    "session_id": session_id,
                    "content": doc.page_content,
                    "file_name": file_name,
                    "metadata": meta,
                    "embedding": emb,
                }
            )
        if rows:
            sb.table("vector_documents").insert(rows).execute()


def process_pdfs_to_supabase(
    sb: Client,
    session_id: str,
    uploaded_files: list,
    embeddings: OpenAIEmbeddings,
) -> tuple[list[str], str | None]:
    all_docs: list[Document] = []
    names: list[str] = []
    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix or ".pdf"
        names.append(uploaded.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            path = tmp.name
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata = dict(d.metadata)
                d.metadata["file_name"] = uploaded.name
            all_docs.extend(docs)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    if not all_docs or not any(d.page_content.strip() for d in all_docs):
        return names, "PDF에서 텍스트를 추출하지 못했습니다."

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    by_file: dict[str, list[Document]] = {}
    for d in all_docs:
        fn = d.metadata.get("file_name") or (names[0] if names else "unknown.pdf")
        by_file.setdefault(fn, []).append(d)

    ensure_session_row(sb, session_id, title="새 세션")
    for fn, file_docs in by_file.items():
        splits = splitter.split_documents(file_docs)
        if not splits:
            continue
        for d in splits:
            d.metadata = dict(d.metadata)
            d.metadata["file_name"] = fn
        insert_vectors_for_files(sb, session_id, splits, fn, embeddings, batch_size=10)

    return names, None


def copy_session_snapshot(
    sb: Client,
    source_session_id: str,
    new_session_id: str,
    title: str,
    chat_history: list[dict[str, str]],
) -> None:
    ensure_session_row(sb, new_session_id, title=title)
    save_messages(sb, new_session_id, chat_history)

    page_size = 500
    offset = 0
    while True:
        r = (
            sb.table("vector_documents")
            .select("content,file_name,metadata,embedding")
            .eq("session_id", source_session_id)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = r.data or []
        if not batch:
            break
        rows = []
        for row in batch:
            rows.append(
                {
                    "session_id": new_session_id,
                    "content": row["content"],
                    "file_name": row["file_name"],
                    "metadata": row.get("metadata") or {},
                    "embedding": row["embedding"],
                }
            )
        if rows:
            sb.table("vector_documents").insert(rows).execute()
        if len(batch) < page_size:
            break
        offset += page_size


def delete_session(sb: Client, session_id: str) -> None:
    sb.table("chat_sessions").delete().eq("id", session_id).execute()


def list_vector_filenames(sb: Client, session_id: str) -> list[str]:
    r = sb.table("vector_documents").select("file_name").eq("session_id", session_id).execute()
    seen: set[str] = set()
    out: list[str] = []
    for row in r.data or []:
        fn = row.get("file_name") or ""
        if fn and fn not in seen:
            seen.add(fn)
            out.append(fn)
    return sorted(out)


def auto_save_session(
    sb: Client,
    session_id: str,
    chat_history: list[dict[str, str]],
    *,
    title: str | None = None,
) -> None:
    ensure_session_row(sb, session_id, title=title or "새 세션")
    save_messages(sb, session_id, chat_history)
    if title:
        sb.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()


def maybe_update_title_from_first_turn(
    sb: Client, session_id: str, chat_history: list[dict[str, str]], llm: ChatOpenAI
) -> None:
    if len(chat_history) < 2:
        return
    first_u = next((m for m in chat_history if m.get("role") == "user"), None)
    first_a = next((m for m in chat_history if m.get("role") == "assistant"), None)
    if not first_u or not first_a:
        return
    if st.session_state.get("title_generated"):
        return
    try:
        t = generate_short_title(llm, first_u["content"], first_a["content"])
        st.session_state.title_generated = True
        sb.table("chat_sessions").update({"title": t}).eq("id", session_id).execute()
    except Exception as e:
        _logger.warning("제목 자동 생성 실패: %s", e)


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "chat_history": [],
        "conversation_memory": deque(maxlen=50),
        "processed_file_names": [],
        "working_session_id": str(uuid.uuid4()),
        "title_generated": False,
        "session_list_rows": [],
        "rag_enabled": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_header() -> None:
    logo_path = _REPO_ROOT / "logo.png"
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if logo_path.is_file():
            st.image(str(logo_path), width=180)
        else:
            st.markdown('<p style="font-size:4rem;">📚</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(
            """
<div style="text-align:center;font-size:4rem !important;line-height:1.2;">
  <span style="color:#1f77b4 !important;">멀티세션</span>
  <span style="color:#ffd700 !important;">RAG 챗봇</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with col3:
        st.empty()

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div[data-testid="stChatMessage"] { padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; }
.stButton > button {
  background-color: #ff69b4 !important;
  color: #fff !important;
  border: none;
}
</style>
""",
        unsafe_allow_html=True,
    )


def load_session_into_state(sb: Client, session_id: str) -> None:
    msgs = load_session_messages(sb, session_id)
    st.session_state.chat_history = msgs
    st.session_state.conversation_memory = deque(list(msgs)[-50:], maxlen=50)
    st.session_state.working_session_id = session_id
    st.session_state.title_generated = True
    try:
        st.session_state.processed_file_names = list_vector_filenames(sb, session_id)
    except Exception:
        st.session_state.processed_file_names = []


def on_session_dropdown_change() -> None:
    sb = st.session_state.get("_supabase_client")
    rows: list = st.session_state.get("session_list_rows") or []
    if not sb or not rows:
        return
    idx = st.session_state.get("session_pick_idx")
    if idx is None or idx >= len(rows):
        return
    sid = str(rows[idx]["id"])
    load_session_into_state(sb, sid)


def main() -> None:
    st.set_page_config(page_title="멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
    init_session_state()
    render_header()

    env = env_status()
    missing = [k for k, v in env.items() if not v]
    sb = get_supabase()

    if missing:
        if _ENV_PATH.is_file():
            _hint = f"로컬 `.env` 경로: {_ENV_PATH}"
        else:
            _hint = (
                "Streamlit Cloud → 앱 설정 → Secrets에 "
                + ", ".join(_SECRET_ENV_KEYS)
                + " 키를 추가하세요."
            )
        st.warning("다음 값이 비어 있습니다: " + ", ".join(missing) + ". " + _hint)

    if sb is None and not missing:
        st.error("Supabase 클라이언트를 만들 수 없습니다. SUPABASE_URL / SUPABASE_ANON_KEY를 확인하세요.")

    if sb is not None:
        st.session_state._supabase_client = sb

    with st.sidebar:
        st.markdown("### 세션 관리")
        rag_on = st.radio("RAG (PDF 검색)", ("RAG 사용", "사용 안 함"), index=0, key="rag_mode")
        st.session_state.rag_enabled = rag_on == "RAG 사용"

        if sb:
            try:
                st.session_state.session_list_rows = list_sessions(sb)
            except Exception as e:
                st.error(f"세션 목록 로드 실패: {e}")
                st.session_state.session_list_rows = []

        rows = st.session_state.session_list_rows
        labels: list[str] = []
        for r in rows:
            tid = str(r.get("id", ""))[:8]
            labels.append(f"{r.get('title', '제목 없음')} ({tid}…)")

        if rows:
            st.selectbox(
                "저장된 세션",
                range(len(rows)),
                format_func=lambda i: labels[i],
                key="session_pick_idx",
                on_change=on_session_dropdown_change,
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("세션로드", use_container_width=True) and sb and rows:
                idx = int(st.session_state.get("session_pick_idx") or 0)
                sid = str(rows[idx]["id"])
                load_session_into_state(sb, sid)
                st.success("세션을 불러왔습니다.")
                st.rerun()
        with c2:
            if st.button("세션저장", use_container_width=True) and sb:
                hist = st.session_state.chat_history
                if len(hist) < 2:
                    st.warning("첫 질문과 답변이 있어야 세션을 저장할 수 있습니다.")
                else:
                    try:
                        llm = get_llm(0.3)
                        u = next((m["content"] for m in hist if m["role"] == "user"), "")
                        a = next((m["content"] for m in hist if m["role"] == "assistant"), "")
                        new_id = str(uuid.uuid4())
                        title = generate_short_title(llm, u, a)
                        copy_session_snapshot(
                            sb,
                            st.session_state.working_session_id,
                            new_id,
                            title,
                            hist,
                        )
                        st.success(f"새 세션으로 저장했습니다: {title}")
                        st.session_state.session_list_rows = list_sessions(sb)
                    except Exception as e:
                        st.error(str(e))
                        _logger.warning("세션 저장 실패: %s", e)

        c3, c4 = st.columns(2)
        with c3:
            if st.button("세션삭제", use_container_width=True) and sb and rows:
                idx = int(st.session_state.get("session_pick_idx") or 0)
                sid = str(rows[idx]["id"])
                try:
                    delete_session(sb, sid)
                    if sid == st.session_state.working_session_id:
                        st.session_state.chat_history = []
                        st.session_state.conversation_memory = deque(maxlen=50)
                        st.session_state.working_session_id = str(uuid.uuid4())
                        st.session_state.title_generated = False
                    st.session_state.session_list_rows = list_sessions(sb)
                    st.success("세션을 삭제했습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        with c4:
            if st.button("화면초기화", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.conversation_memory = deque(maxlen=50)
                st.session_state.processed_file_names = []
                st.session_state.working_session_id = str(uuid.uuid4())
                st.session_state.title_generated = False
                st.rerun()

        if st.button("vectordb", use_container_width=True) and sb:
            names = list_vector_filenames(sb, st.session_state.working_session_id)
            if names:
                st.info("현재 작업 세션에 연결된 파일명:\n\n" + "\n".join(f"- {n}" for n in names))
            else:
                st.info("현재 세션에 저장된 벡터 문서가 없습니다.")

        st.markdown("---")
        uploads = st.file_uploader("PDF 파일 업로드", type=["pdf"], accept_multiple_files=True)
        if st.button("파일 처리하기") and sb:
            if not uploads:
                st.warning("PDF 파일을 선택해 주세요.")
            elif not env["OPENAI_API_KEY"]:
                st.warning("OPENAI_API_KEY가 없어 임베딩을 만들 수 없습니다.")
            else:
                with st.spinner("PDF 처리 중..."):
                    try:
                        emb = get_embeddings()
                        ensure_session_row(sb, st.session_state.working_session_id, title="새 세션")
                        names, err = process_pdfs_to_supabase(
                            sb, st.session_state.working_session_id, list(uploads), emb
                        )
                        if err:
                            st.error(err)
                        else:
                            st.session_state.processed_file_names = names
                            st.success(f"처리 완료: {len(names)}개 파일 (세션에 저장됨)")
                            auto_save_session(sb, st.session_state.working_session_id, st.session_state.chat_history)
                    except Exception as e:
                        st.error(str(e))
                        _logger.warning("PDF 처리 오류: %s", e)

        if st.session_state.processed_file_names:
            st.caption("처리된 파일")
            for fn in st.session_state.processed_file_names:
                st.caption(f"- {fn}")

        st.text(
            f"모델: {LLM_MODEL}\n"
            f"작업 세션 ID: {st.session_state.working_session_id[:8]}…\n"
            f"메시지 수: {len(st.session_state.chat_history)}"
        )

    user_input = st.chat_input("메시지를 입력하세요")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.conversation_memory.append({"role": "user", "content": user_input})

    for msg in st.session_state.chat_history:
        role = msg.get("role", "user")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""), unsafe_allow_html=False)

    if not user_input:
        return

    if not env["OPENAI_API_KEY"]:
        with st.chat_message("assistant"):
            st.markdown(
                "# 안내\n\n## 설정\n\n### 안내\n\nOPENAI_API_KEY가 없어 응답을 생성할 수 없습니다."
            )
        return

    with st.chat_message("assistant"):
        out = st.empty()
        main_text = ""
        try:
            llm = get_llm()
            follow_llm = get_llm(0.3)

            ws = st.session_state.working_session_id
            use_rag = st.session_state.rag_enabled and sb is not None

            if use_rag:
                emb = get_embeddings()
                mem_list = list(st.session_state.conversation_memory)
                main_core = stream_rag_answer(
                    llm, sb, ws, emb, user_input, mem_list[:-1], out
                )
            else:
                mem_list = list(st.session_state.conversation_memory)
                hist_msgs: list[BaseMessage] = [SystemMessage(content=ANSWER_STYLE)]
                for m in mem_list[:-1]:
                    if m["role"] == "user":
                        hist_msgs.append(HumanMessage(content=m["content"]))
                    else:
                        hist_msgs.append(AIMessage(content=m["content"]))
                hist_msgs.append(HumanMessage(content=user_input))
                main_core = stream_direct_llm(llm, hist_msgs, out)

            sug = generate_followup_questions(follow_llm, user_input, main_core)
            main_text = remove_separators(main_core + sug)
            out.markdown(main_text)

        except ValueError as e:
            main_text = f"# 안내\n\n## 설정\n\n### 안내\n\n{e}"
            out.markdown(main_text)
        except Exception as e:
            _logger.warning("응답 생성 오류: %s", e, exc_info=True)
            main_text = (
                "# 오류\n\n## 응답을 만들 수 없습니다\n\n"
                f"### 안내\n\n일시적인 오류가 발생했습니다. ({e})"
            )
            out.markdown(main_text)

        final_content = remove_separators(main_text)
        st.session_state.chat_history.append({"role": "assistant", "content": final_content})
        st.session_state.conversation_memory.append({"role": "assistant", "content": final_content})

        if sb:
            try:
                ensure_session_row(sb, st.session_state.working_session_id, title="새 세션")
                maybe_update_title_from_first_turn(
                    sb, st.session_state.working_session_id, st.session_state.chat_history, get_llm(0.3)
                )
                trow = (
                    sb.table("chat_sessions")
                    .select("title")
                    .eq("id", st.session_state.working_session_id)
                    .limit(1)
                    .execute()
                )
                cur_title = (trow.data[0]["title"] if trow.data else None) or "새 세션"
                auto_save_session(
                    sb,
                    st.session_state.working_session_id,
                    st.session_state.chat_history,
                    title=cur_title,
                )
                st.session_state.session_list_rows = list_sessions(sb)
            except Exception as e:
                _logger.warning("자동 저장 실패: %s", e)


if __name__ == "__main__":
    main()
