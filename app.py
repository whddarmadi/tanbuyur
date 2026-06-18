import os
import json
import time
import datetime
import hashlib
import re
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langdetect import detect

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE        = "🌱 Asisten Budidaya Sayuran"
APP_SUBTITLE     = "Panduan Teknis Berbasis AI"
COLLECTION_NAME  = "sayuran_kb"
EMBED_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM        = 384
GROQ_MODEL       = "llama-3.1-8b-instant"
TOP_K            = 5
MIN_RELEVANCE    = 0.35
RAG_SCORE_THRESHOLD = 0.50   # Di bawah ini = out-of-domain, tidak panggil LLM
MAX_MEMORY_TURNS = 6
LOG_FILE         = "logs/chat_log.jsonl"

# ── Katalog 30 sayuran (hardcoded, hemat token) ───────────────────────────────
KNOWLEDGE_BASE_CATALOG = """Kelompok Bawang    : Bawang Daun, Bawang Merah, Bawang Putih
Sayuran Daun       : Bayam, Caisin, Kailan (Kaelan), Kangkung, Katuk, Pakchoi, Petsai, Selada, Seledri
Sayuran Buah       : Cabai Merah, Cabai Rawit, Kacang Panjang, Labu Siam, Mentimun, Oyong, Paria, Terung, Tomat
Sayuran Umbi       : Kentang, Wortel
Kubis-kubisan      : Brokoli, Kubis, Kubis Bunga
Kacang-kacangan    : Buncis, Kapri (Ercis)
Jamur              : Jamur Merang, Jamur Kayu (Tiram, Kuping, Shiitake)"""

# ── Set untuk validasi Python (poin 1 GPT) ────────────────────────────────────
KNOWN_PLANTS = {
    # bawang
    "bawang daun", "bawang merah", "bawang putih",
    # daun
    "bayam", "caisin", "kailan", "kaelan", "kangkung", "katuk",
    "pakchoi", "pakcoy", "petsai", "selada", "seledri",
    # buah
    "cabai", "cabai merah", "cabai rawit", "cabe", "kacang panjang",
    "labu siam", "mentimun", "timun", "oyong", "paria", "pare",
    "terung", "terong", "tomat",
    # umbi
    "kentang", "wortel",
    # kubis
    "brokoli", "kubis", "kubis bunga", "kembang kol",
    # kacang
    "buncis", "kapri", "ercis",
    # jamur
    "jamur", "jamur merang", "jamur kayu", "jamur tiram",
    "jamur kuping", "shiitake",
}

# ── Query expansion — sinonim sehari-hari → istilah teknis buku (poin 4) ──────
QUERY_SYNONYMS = {
    "nanam":      "budidaya penanaman",
    "tanam":      "budidaya penanaman",
    "menanam":    "budidaya penanaman",
    "tanamkan":   "penanaman",
    "hama":       "organisme pengganggu tumbuhan OPT",
    "serangga":   "hama organisme pengganggu",
    "penyakit":   "penyakit tanaman patogen",
    "pupuk":      "pemupukan nutrisi",
    "siram":      "penyiraman irigasi",
    "panen":      "pemanenan panen",
    "bibit":      "benih pembibitan",
    "tanah":      "media tanam tanah",
    "pot":        "wadah polybag media tanam",
    "tumbuh":     "pertumbuhan",
    "mati":       "kematian tanaman gangguan",
    "kuning":     "klorosis gejala penyakit",
    "layu":       "kelayuan penyakit fusarium",
    "busuk":      "pembusukan penyakit",
    "berbuah":    "pembuahan produktivitas",
}

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
    .stApp { background-color: #f7fdf4; }
    .main .block-container { padding-top: 1.5rem; max-width: 900px; }

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
    .score-high { color: #1b5e20; font-weight: 700; }
    .score-mid  { color: #f57f17; font-weight: 700; }
    .score-low  { color: #b71c1c; font-weight: 700; }

    .mode-rag  { background:#e8f5e9; color:#1b5e20; border:1px solid #81c784;
                 border-radius:10px; padding:2px 10px; font-size:0.75rem; font-weight:700; }
    .mode-oos  { background:#fdecea; color:#b71c1c; border:1px solid #ef9a9a;
                 border-radius:10px; padding:2px 10px; font-size:0.75rem; font-weight:700; }
    .mode-list { background:#e3f2fd; color:#1565c0; border:1px solid #90caf9;
                 border-radius:10px; padding:2px 10px; font-size:0.75rem; font-weight:700; }

    section[data-testid="stSidebar"] { background: #e8f5e9; }
    [data-testid="stMetricValue"] { color: #2e7d32; font-size: 1.3rem !important; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE LOADERS (cached)
# ─────────────────────────────────────────────────────────────────────────────
def get_groq_client() -> Groq:
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY belum diset di Streamlit Secrets.")
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner="🔗 Menghubungkan ke Qdrant Cloud...")
def load_qdrant_client() -> QdrantClient | None:
    url     = st.secrets.get("QDRANT_URL")     or os.environ.get("QDRANT_URL")
    api_key = st.secrets.get("QDRANT_API_KEY") or os.environ.get("QDRANT_API_KEY")
    if not url or not api_key:
        return None
    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=20)
        client.get_collection(COLLECTION_NAME)
        return client
    except Exception as e:
        st.error(f"❌ Gagal konek ke Qdrant: {e}")
        return None


@st.cache_resource(show_spinner="🧠 Memuat embedding model...")
def load_embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "id"


def expand_query(query: str) -> str:
    """Tambahkan sinonim teknis ke query untuk meningkatkan retrieval (poin 4)."""
    q_lower = query.lower()
    expansions = []
    for casual, technical in QUERY_SYNONYMS.items():
        if casual in q_lower:
            expansions.append(technical)
    if expansions:
        return query + " " + " ".join(expansions)
    return query


def detect_plant(query: str) -> str | None:
    """
    Deteksi nama tanaman dalam query (poin 1).
    Return nama tanaman jika ditemukan, None jika tidak.
    """
    q_lower = query.lower()
    for plant in KNOWN_PLANTS:
        if plant in q_lower:
            return plant
    return None


def is_catalog_query(query: str) -> bool:
    """Deteksi query yang meminta daftar isi / katalog sayuran."""
    keywords = [
        "daftar", "list", "apa saja", "semua", "sebutkan",
        "database", "knowledge base", "katalog", "isi buku",
        "daftar isi", "tanaman apa", "sayuran apa"
    ]
    q_lower = query.lower()
    return any(kw in q_lower for kw in keywords)


def retrieve_context(
    qdrant: QdrantClient,
    embed_model: SentenceTransformer,
    query: str,
    top_k: int = TOP_K
) -> list[dict]:
    expanded = expand_query(query)
    query_vector = embed_model.encode(expanded, normalize_embeddings=True).tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        score_threshold=MIN_RELEVANCE,
    )
    chunks = []
    for hit in results:
        chunks.append({
            "text":      hit.payload.get("text", ""),
            "page_num":  hit.payload.get("page_num", "?"),
            "chunk_idx": hit.payload.get("chunk_idx", 0),
            "score":     round(hit.score, 4),
            "lang":      hit.payload.get("lang", "id"),
        })
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def build_context_string(chunks: list[dict]) -> str:
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
    relevant = history[-max_turns * 2:] if len(history) > max_turns * 2 else history
    lines = []
    for msg in relevant:
        role = "Pengguna" if msg["role"] == "user" else "Asisten"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def call_llm(
    client: Groq, query: str, context: str, memory: str, lang: str
) -> tuple[str, int, int]:
    lang_instruction = (
        "Jawablah dalam Bahasa Indonesia."
        if lang == "id"
        else "Answer in the same language as the question."
    )
    system_prompt = f"""Kamu adalah asisten pertanian yang ramah dan ahli, \
khusus membantu petani dan pekebun memahami teknik budidaya tanaman sayuran.

DATABASE mencakup tepat 30 jenis sayuran:
{KNOWLEDGE_BASE_CATALOG}

Tugasmu:
1. Jawab pertanyaan berdasarkan KONTEKS dokumen yang diberikan.
2. Gunakan bahasa yang mudah dipahami — hindari jargon tanpa penjelasan.
3. Bila relevan, berikan langkah-langkah praktis yang bisa langsung dijalankan.
4. {lang_instruction}

Format jawaban:
- Langsung ke inti jawaban
- Gunakan poin-poin jika ada beberapa langkah
- Akhiri dengan satu kalimat tips atau motivasi singkat jika sesuai"""

    user_message = f"""RIWAYAT PERCAKAPAN:
{memory if memory else "(Tidak ada riwayat)"}

KONTEKS DARI DOKUMEN:
{context}

PERTANYAAN SAAT INI:
{query}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.4,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content
    usage  = response.usage
    return answer, usage.prompt_tokens, usage.completion_tokens


def score_color_class(score: float) -> str:
    if score >= 0.7:  return "score-high"
    if score >= 0.5:  return "score-mid"
    return "score-low"


def save_log(
    session_id: str, query: str, answer: str,
    chunks: list[dict], tokens: dict,
    answer_mode: str, plant_detected: str | None
):
    """Log diperkaya dengan answer_mode dan plant_detected (poin 5)."""
    os.makedirs("logs", exist_ok=True)
    entry = {
        "ts":               datetime.datetime.now().isoformat(),
        "session_id":       session_id,
        "query":            query,
        "answer":           answer,
        "answer_mode":      answer_mode,
        "plant_detected":   plant_detected,
        "retrieval_success": answer_mode == "RAG",
        "top_chunk_score":  chunks[0]["score"] if chunks else None,
        "chunks_used":      len(chunks),
        "sources":          [{"page": c["page_num"], "score": c["score"]} for c in chunks],
        "tokens":           tokens,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_log_summary() -> dict:
    if not os.path.exists(LOG_FILE):
        return {"total": 0, "avg_score": 0, "total_tokens": 0, "oos_pct": 0}
    entries = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    if not entries:
        return {"total": 0, "avg_score": 0, "total_tokens": 0, "oos_pct": 0}
    scores    = [e["top_chunk_score"] for e in entries if e.get("top_chunk_score")]
    total_tok = sum(e.get("tokens", {}).get("total", 0) for e in entries)
    oos_count = sum(1 for e in entries if e.get("answer_mode") == "OUT_OF_SCOPE")
    return {
        "total":        len(entries),
        "avg_score":    round(sum(scores) / len(scores) * 100, 1) if scores else 0,
        "total_tokens": total_tok,
        "oos_pct":      round(oos_count / len(entries) * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "session_id"    not in st.session_state: st.session_state.session_id    = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "total_queries" not in st.session_state: st.session_state.total_queries = 0
if "total_tokens"  not in st.session_state: st.session_state.total_tokens  = 0


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
groq_client = get_groq_client()
qdrant      = load_qdrant_client()
embed_model = load_embed_model()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 Asisten Budidaya Sayuran")
    st.markdown("---")

    st.markdown("**📦 Vector Database**")
    if qdrant:
        try:
            info  = qdrant.get_collection(COLLECTION_NAME)
            count = info.points_count
            st.success(f"✅ Qdrant Cloud — {count:,} chunk")
        except Exception:
            st.warning("⚠️ Terkoneksi tapi collection belum ada")
    else:
        st.error("❌ Qdrant tidak terhubung")
        st.markdown("Set `QDRANT_URL` dan `QDRANT_API_KEY` di Streamlit Secrets.")

    st.markdown("---")
    st.markdown("**🤖 Model & Konfigurasi**")
    st.markdown(f"- LLM: `{GROQ_MODEL}`")
    st.markdown(f"- Embedding: `paraphrase-multilingual`")
    st.markdown(f"- Top-K: `{TOP_K}` chunk")
    st.markdown(f"- Min Relevansi: `{int(MIN_RELEVANCE*100)}%`")
    st.markdown(f"- RAG Threshold: `{int(RAG_SCORE_THRESHOLD*100)}%`")
    st.markdown(f"- Memory: `{MAX_MEMORY_TURNS}` turn")

    st.markdown("---")
    st.markdown("**📊 Sesi Ini**")
    col1, col2 = st.columns(2)
    col1.metric("💬 Query",  st.session_state.total_queries)
    col2.metric("🔤 Token",  f"{st.session_state.total_tokens:,}")

    st.markdown("---")
    st.markdown("**📋 Statistik Log**")
    log_stats = load_log_summary()
    st.markdown(f"- Total query: **{log_stats['total']}**")
    st.markdown(f"- Avg relevansi: **{log_stats['avg_score']}%**")
    st.markdown(f"- Total token: **{log_stats['total_tokens']:,}**")
    st.markdown(f"- Query di luar DB: **{log_stats['oos_pct']}%**")

    st.markdown("---")
    if st.button("🗑️ Hapus Riwayat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
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

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">🧑‍🌾 <strong>Kamu:</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        # Mode badge (poin 3)
        mode       = msg.get("answer_mode", "RAG")
        mode_label = {
            "RAG":          "🟢 Berdasarkan Dokumen",
            "OUT_OF_SCOPE": "🔴 Tidak Ditemukan di Database",
            "CATALOG":      "🔵 Daftar Isi Knowledge Base",
        }.get(mode, "🟢 Berdasarkan Dokumen")
        mode_class = {
            "RAG":          "mode-rag",
            "OUT_OF_SCOPE": "mode-oos",
            "CATALOG":      "mode-list",
        }.get(mode, "mode-rag")

        st.markdown(
            f'<div class="bot-msg">🌱 <strong>Asisten:</strong> '
            f'<span class="{mode_class}">{mode_label}</span><br><br>'
            f'{msg["content"]}</div>',
            unsafe_allow_html=True
        )
        if msg.get("sources"):
            badges = " ".join([
                f'<span class="source-badge">📄 Hal. {s["page"]} '
                f'<span class="{score_color_class(s["score"])}">{round(s["score"]*100,1)}%</span></span>'
                for s in msg["sources"]
            ])
            st.markdown(
                f'<div style="margin:-4px 0 12px 0;padding-left:8px;">{badges}</div>',
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_query", "")
query   = st.chat_input("Tanya tentang budidaya sayuran... 🌿") or prefill

if query:
    if not qdrant:
        st.error("❌ Qdrant belum terhubung. Cek Streamlit Secrets.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🌱 Mencari informasi dan menyusun jawaban..."):
        lang          = detect_language(query)
        plant         = detect_plant(query)
        answer_mode   = "RAG"
        prompt_tok    = 0
        comp_tok      = 0
        chunks        = []

        # ── Poin 3: Deteksi query katalog dulu ───────────────────────────────
        if is_catalog_query(query) and not plant:
            answer      = f"Berikut adalah daftar lengkap sayuran dalam database Tanbuyur:\n\n{KNOWLEDGE_BASE_CATALOG}\n\nSilakan tanyakan cara budidaya, hama, pupuk, atau panen untuk sayuran mana saja dari daftar di atas! 🌱"
            answer_mode = "CATALOG"

        else:
            # ── Poin 1: Validasi Python — cek tanaman dikenal ────────────────
            chunks = retrieve_context(qdrant, embed_model, query)

            # ── Poin 2 & 6: Cek skor threshold — kalau rendah, stop di sini ─
            if not chunks or chunks[0]["score"] < RAG_SCORE_THRESHOLD:
                plant_name = plant or "tanaman tersebut"
                answer = (
                    f"Maaf, informasi tentang **{plant_name}** tidak tersedia "
                    f"dalam database Tanbuyur.\n\n"
                    f"Database Tanbuyur hanya mencakup **30 jenis sayuran** berikut:\n\n"
                    f"{KNOWLEDGE_BASE_CATALOG}\n\n"
                    f"Silakan tanyakan tentang salah satu sayuran di atas! 🌿"
                )
                answer_mode = "OUT_OF_SCOPE"
                chunks      = []  # tidak tampilkan source badge

            else:
                # ── RAG normal — panggil LLM ──────────────────────────────────
                context  = build_context_string(chunks)
                memory   = build_memory_string(st.session_state.messages[:-1])
                answer, prompt_tok, comp_tok = call_llm(
                    groq_client, query, context, memory, lang
                )
                answer_mode = "RAG"

    total_tok = prompt_tok + comp_tok
    st.session_state.total_queries += 1
    st.session_state.total_tokens  += total_tok

    sources = [{"page": c["page_num"], "score": c["score"]} for c in chunks]
    st.session_state.messages.append({
        "role":        "assistant",
        "content":     answer,
        "sources":     sources,
        "answer_mode": answer_mode,
    })

    save_log(
        session_id=st.session_state.session_id,
        query=query,
        answer=answer,
        chunks=chunks,
        tokens={"prompt": prompt_tok, "completion": comp_tok, "total": total_tok},
        answer_mode=answer_mode,
        plant_detected=plant,
    )
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:40px 20px; color:#558b2f;">
        <div style="font-size:4rem;">🌿</div>
        <h3>Halo, Petani!</h3>
        <p>Tanya apapun tentang budidaya tanaman sayuran.<br>
        Aku siap membantu berdasarkan petunjuk teknis resmi.</p>
        <p style="font-size:0.85rem;color:#81c784;">
            Coba klik salah satu contoh pertanyaan di sidebar →
        </p>
    </div>
    """, unsafe_allow_html=True)
