# 🌱 Asisten Budidaya Sayuran — RAG Chatbot

Chatbot pertanian berbasis **Retrieval-Augmented Generation (RAG)** yang menjawab pertanyaan teknis budidaya sayuran dari dokumen PDF resmi.

---

## Fitur Utama

| Fitur | Detail |
|---|---|
| 📚 Sumber pengetahuan | Petunjuk Teknis Budidaya Tanaman Sayuran (PDF) |
| 🔍 Retrieval | ChromaDB + cosine similarity scoring |
| 🌐 Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (Indonesia & English) |
| 🤖 LLM | Groq — Llama 3.1 8B Instant |
| 🧠 Multi-turn memory | 6 turn terakhir diingat per sesi |
| 📊 Relevance scoring | Skor per chunk ditampilkan di UI |
| 📋 Chat log | Disimpan ke `logs/chat_log.jsonl` |
| 🚀 Deployment | Streamlit Cloud |

---

## Struktur Proyek

```
tanbuyur/
├── app.py                        # Aplikasi Streamlit utama
├── ingestion_pipeline.ipynb      # Notebook Colab untuk ingest PDF ke ChromaDB
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example     # Template konfigurasi API key
├── chroma_db/                    # ← Dibuat oleh notebook (tidak di-git)
│   └── ...
└── logs/
    └── chat_log.jsonl            # ← Dibuat otomatis saat ada query
```

---

## Cara Pakai

### 1. Ingest PDF (Google Colab)

Buka `ingestion_pipeline.ipynb` di Colab, lalu:

1. Jalankan semua cell secara berurutan
2. Upload file PDF saat diminta
3. Tunggu proses chunking + embedding (±3–5 menit untuk PDF 100 hal.)
4. Download folder `chroma_db/` dari Colab Files atau Google Drive

### 2. Setup Lokal

```bash
# Clone repo
git clone https://github.com/<username>/tanbuyur.git
cd tanbuyur

# Install dependencies
pip install -r requirements.txt

# Copy template secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# → Edit secrets.toml, isi GROQ_API_KEY kamu

# Letakkan folder chroma_db/ di root direktori ini

# Jalankan
streamlit run app.py
```

### 3. Deploy ke Streamlit Cloud

1. Push repo ke GitHub (pastikan `chroma_db/` dan `.streamlit/secrets.toml` ada di `.gitignore`)
2. Buka [share.streamlit.io](https://share.streamlit.io) → New App → pilih repo
3. Di **Settings > Secrets**, tambahkan:
   ```toml
   GROQ_API_KEY = "gsk_xxxx..."
   ```
4. Upload `chroma_db/` ke repo **atau** mount via Google Drive di notebook

> **Catatan:** Karena ChromaDB menyimpan file lokal, untuk Streamlit Cloud kamu perlu commit folder `chroma_db/` ke repo (ukurannya kecil, biasanya <50MB untuk buku teknis).

---

## Parameter Konfigurasi

Semua parameter utama ada di bagian `CONFIG` di `app.py`:

```python
TOP_K = 5                    # Jumlah chunk yang diambil saat retrieval
MIN_RELEVANCE_SCORE = 0.35   # Threshold minimum skor cosine (0–1)
MAX_MEMORY_TURNS = 6         # Jumlah turn percakapan yang disimpan sebagai memori
GROQ_MODEL = "llama-3.1-8b-instant"
```

---

## Format Log

Setiap query disimpan ke `logs/chat_log.jsonl`:

```json
{
  "ts": "2024-11-15T14:32:01.123456",
  "session_id": "a1b2c3d4",
  "query": "cara menanam cabai",
  "answer": "...",
  "top_chunk_score": 0.82,
  "chunks_used": 4,
  "sources": [
    {"page": 23, "score": 0.82},
    {"page": 24, "score": 0.76}
  ],
  "tokens": {"prompt": 1204, "completion": 312, "total": 1516}
}
```

---

## Embedding Model

Model `paraphrase-multilingual-MiniLM-L12-v2` dipilih karena:
- Mendukung 50+ bahasa termasuk Indonesia
- Ukuran ringan (~118 MB) — cocok untuk Colab free tier
- Performa retrieval baik untuk teks campuran Indonesia-Latin pertanian
- Tersedia langsung via `sentence-transformers`

---

## Dependensi Utama

```
groq                         # Client Groq API
chromadb                     # Vector store lokal
sentence-transformers        # Multilingual embedding
pypdf + pdfplumber           # Ekstraksi teks PDF
streamlit                    # Web interface
langdetect                   # Deteksi bahasa otomatis
python-dotenv                # Manajemen environment variable
```

---

## Lisensi

Proyek pribadi — bebas digunakan dan dimodifikasi.
