import os
import json
import time
import datetime
import hashlib
import streamlit as st
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from langdetect import detect

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE = "🌱 Asisten Budidaya Sayuran"
APP_SUBTITLE = "Panduan Teknis Berbasis AI"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "sayuran_kb"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5                   # jumlah chunk yang diambil dari retrieval
MIN_RELEVANCE_SCORE = 0.35  # threshold cosine similarity (0–1)
MAX_MEMORY_TURNS = 6        # jumlah turn percakapan yang diingat
LOG_FILE = "logs/chat_log.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main layout */
    .stApp { background-color: #f7fdf4; }
    .main .block-container { padding-top: 1.5rem; max-width: 900px; }

    /* Chat bubbles */
    .user-msg {
        background: #d4edda;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        border-left: 4px solid #28a745;
        font-size: 0.95rem;
    }
    .bot-msg {
        background: #ffffff;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 10%;
        border-left: 4px solid #6aaa64;
        font-size: 0.95rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        background: #e8f5e9;
        border: 1px solid #81c784;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #2e7d32;
        margin: 2px;
    }

    /* Score badge */
    .score-high  { color: #1b5e20; font-weight: 700; }
    .score-mid   { color: #f57f17; font-weight: 700; }
    .score-low   { color: #b71c1c; font-weight: 700; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #e8f5e9; }
    .sidebar-header { font-size: 1.1rem; font-weight: 700; color: #1b5e20; }

    /* Metrics */
    [data-testid="stMetricValue"] { color: #2e7d32; font-size: 1.3rem !important; }

    /* Info / warning box */
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_groq_client() -> Groq:
    """Inisialisasi Groq client dari Streamlit secrets atau env."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY belum diset. Tambahkan ke `.streamlit/secrets.toml` atau environment variable.")
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner="🔗 Memuat vector database...")
def load_chroma_collection():
    """Load ChromaDB collection (cached, hanya sekali per session)."""
    if not os.path.exists(CHROMA_DIR):
        return None
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
        return collection
    except Exception:
        return None


def detect_language(text: str) -> str:
    """Deteksi bahasa teks (id/en/other)."""
    try:
        return detect(text)
    except Exception:
        return "id"


def retrieve_context(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve top-k chunk paling relevan dari ChromaDB.
    Mengembalikan list of dict dengan text, metadata, dan relevance score.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = round(1 - dist, 4)  # cosine: 1 = identik, 0 = tidak relevan
        chunks.append({
            "text": doc,
            "page_num": meta.get("page_num", "?"),
            "chunk_idx": meta.get("chunk_idx", 0),
            "score": score,
            "lang": meta.get("lang", "id"),
        })
    # Filter dan urutkan berdasarkan skor
    filtered = [c for c in chunks if c["score"] >= MIN_RELEVANCE_SCORE]
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered


def build_context_string(chunks: list[dict]) -> str:
    """Gabungkan chunk menjadi context string yang rapi untuk prompt."""
    if not chunks:
        return "[Tidak ada konteks relevan ditemukan]"
    parts = []
    for i, c in enumerate(chunks, 1):
        score_pct = round(c["score"] * 100, 1)
        parts.append(
            f"[Sumber {i} | Hal. {c['page_num']} | Relevansi: {score_pct}%]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def build_memory_string(history: list[dict], max_turns: int = MAX_MEMORY_TURNS) -> str:
    """Ambil N turn terakhir sebagai memory string untuk prompt."""
    relevant = history[-max_turns * 2:] if len(history) > max_turns * 2 else history
    lines = []
    for msg in relevant:
        role = "Pengguna" if msg["role"] == "user" else "Asisten"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def call_llm(client: Groq, query: str, context: str, memory: str, lang: str) -> tuple[str, int, int]:
    """
    Panggil Groq LLM dengan context RAG dan memory percakapan.
    Mengembalikan (answer, prompt_tokens, completion_tokens).
    """
    lang_instruction = "Jawablah dalam Bahasa Indonesia." if lang == "id" else "Answer in the same language as the question."

    system_prompt = f"""Kamu adalah asisten pertanian yang ramah dan ahli, \
khusus membantu petani dan pekebun memahami teknik budidaya tanaman sayuran.

Tugasmu:
1. Jawab pertanyaan berdasarkan KONTEKS yang diberikan.
2. Jika konteks tidak cukup, katakan dengan jujur dan berikan pengetahuan umum pertanian.
3. Gunakan bahasa yang mudah dipahami — hindari jargon tanpa penjelasan.
4. Bila relevan, berikan langkah-langkah praktis yang bisa langsung dijalankan.
5. {lang_instruction}

Format jawaban:
- Langsung ke inti jawaban
- Gunakan poin-poin jika ada beberapa langkah
- Akhiri dengan satu kalimat tips atau motivasi singkat jika sesuai
"""

    user_message = f"""RIWAYAT PERCAKAPAN:
{memory if memory else "(Tidak ada riwayat)"}

KONTEKS DARI DOKUMEN:
{context}

PERTANYAAN SAAT INI:
{query}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.4,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content
    usage = response.usage
    return answer, usage.prompt_tokens, usage.completion_tokens


def score_color_class(score: float) -> str:
    if score >= 0.7:
        return "score-high"
    elif score >= 0.5:
        return "score-mid"
    return "score-low"


def save_log(session_id: str, query: str, answer: str, chunks: list[dict], tokens: dict):
    """Simpan log chat ke file JSONL."""
    os.makedirs("logs", exist_ok=True)
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "query": query,
        "answer": answer,
        "top_chunk_score": chunks[0]["score"] if chunks else None,
        "chunks_used": len(chunks),
        "sources": [{"page": c["page_num"], "score": c["score"]} for c in chunks],
        "tokens": tokens,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_log_summary() -> dict:
    """Baca log dan kembalikan statistik ringkasan."""
    if not os.path.exists(LOG_FILE):
        return {"total": 0, "avg_score": 0, "total_tokens": 0}
    entries = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    if not entries:
        return {"total": 0, "avg_score": 0, "total_tokens": 0}
    scores = [e["top_chunk_score"] for e in entries if e.get("top_chunk_score")]
    total_tok = sum(
        e.get("tokens", {}).get("total", 0) for e in entries
    )
    return {
        "total": len(entries),
        "avg_score": round(sum(scores) / len(scores) * 100, 1) if scores else 0,
        "total_tokens": total_tok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = hashlib.md5(
        str(time.time()).encode()
    ).hexdigest()[:8]

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
groq_client = get_groq_client()
collection = load_chroma_collection()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 Asisten Budidaya Sayuran")
    st.markdown("---")

    # Status vector DB
    st.markdown("**📦 Vector Database**")
    if collection:
        count = collection.count()
        st.success(f"✅ Terhubung — {count:,} chunk")
    else:
        st.error("❌ `chroma_db/` tidak ditemukan")
        st.markdown(
            "Jalankan notebook `ingestion_pipeline.ipynb` "
            "lalu upload folder `chroma_db/` ke direktori ini."
        )

    st.markdown("---")

    # Model info
    st.markdown("**🤖 Model & Konfigurasi**")
    st.markdown(f"- LLM: `{GROQ_MODEL}`")
    st.markdown(f"- Embedding: `paraphrase-multilingual`")
    st.markdown(f"- Top-K: `{TOP_K}` chunk")
    st.markdown(f"- Min Relevansi: `{int(MIN_RELEVANCE_SCORE*100)}%`")
    st.markdown(f"- Memory: `{MAX_MEMORY_TURNS}` turn")

    st.markdown("---")

    # Session stats
    st.markdown("**📊 Sesi Ini**")
    col1, col2 = st.columns(2)
    col1.metric("💬 Query", st.session_state.total_queries)
    col2.metric("🔤 Token", f"{st.session_state.total_tokens:,}")

    st.markdown("---")

    # Log summary
    st.markdown("**📋 Statistik Log**")
    log_stats = load_log_summary()
    st.markdown(f"- Total query: **{log_stats['total']}**")
    st.markdown(f"- Avg relevansi: **{log_stats['avg_score']}%**")
    st.markdown(f"- Total token: **{log_stats['total_tokens']:,}**")

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Hapus Riwayat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Sample questions
    st.markdown("**💡 Contoh Pertanyaan**")
    sample_qs = [
        "Bagaimana cara menanam cabai dengan benar?",
        "Apa penyebab daun tomat menguning?",
        "Berapa kebutuhan air untuk bayam per minggu?",
        "Pupuk apa yang cocok untuk kangkung?",
        "Kapan waktu terbaik panen terong?",
    ]
    for q in sample_qs:
        if st.button(q, key=f"sample_{hash(q)}", use_container_width=True):
            st.session_state["prefill_query"] = q
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"## {APP_TITLE}")
st.markdown(f"*{APP_SUBTITLE}*")
st.markdown("---")

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">🧑‍🌾 <strong>Kamu:</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="bot-msg">🌱 <strong>Asisten:</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )
        # Tampilkan sources jika ada
        if msg.get("sources"):
            badges = " ".join([
                f'<span class="source-badge">'
                f'📄 Hal. {s["page"]} '
                f'<span class="{score_color_class(s["score"])}">'
                f'{round(s["score"]*100,1)}%</span></span>'
                for s in msg["sources"]
            ])
            st.markdown(
                f'<div style="margin: -4px 0 12px 0; padding-left: 8px;">{badges}</div>',
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_query", "")
query = st.chat_input(
    "Tanya tentang budidaya sayuran... 🌿",
    key="chat_input"
) or prefill

if query:
    if not collection:
        st.error("❌ Vector database belum tersedia. Jalankan ingestion notebook terlebih dahulu.")
        st.stop()

    # Tambah pesan user ke history
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🌱 Mencari informasi dan menyusun jawaban..."):
        # 1. Deteksi bahasa
        lang = detect_language(query)

        # 2. Retrieve dari ChromaDB
        chunks = retrieve_context(collection, query)

        # 3. Build context dan memory
        context = build_context_string(chunks)
        memory = build_memory_string(st.session_state.messages[:-1])

        # 4. Panggil LLM
        answer, prompt_tok, comp_tok = call_llm(
            groq_client, query, context, memory, lang
        )
        total_tok = prompt_tok + comp_tok

    # Update stats
    st.session_state.total_queries += 1
    st.session_state.total_tokens += total_tok

    # Simpan jawaban ke history dengan metadata sources
    sources = [{"page": c["page_num"], "score": c["score"]} for c in chunks]
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    # Simpan log
    save_log(
        session_id=st.session_state.session_id,
        query=query,
        answer=answer,
        chunks=chunks,
        tokens={"prompt": prompt_tok, "completion": comp_tok, "total": total_tok}
    )

    # Rerun untuk render ulang chat
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 40px 20px; color: #558b2f;">
        <div style="font-size: 4rem;">🌿</div>
        <h3>Halo, Petani!</h3>
        <p>Tanya apapun tentang budidaya tanaman sayuran.<br>
        Aku siap membantu berdasarkan petunjuk teknis resmi.</p>
        <p style="font-size: 0.85rem; color: #81c784;">
            Coba klik salah satu contoh pertanyaan di sidebar →
        </p>
    </div>
    """, unsafe_allow_html=True)
