# Multi-Agent-Based AI-Powered Exam Pattern Analysis and Question Prediction System

**Advanced Artificial Intelligence — Group Project**

A Streamlit application that analyzes a student's own past exam papers using OCR, sentence embeddings, unsupervised topic clustering, retrieval-augmented generation, and a multi-agent LLM workflow to discover recurring question patterns and generate probable future exam questions.

YouTube Link of the project: https://youtu.be/IjkuVFnx1JY

---

## 1. Problem Statement

Students preparing for exams often revisit years of past papers manually, trying to spot which topics repeat and which questions are likely to reappear. This is slow, subjective, and doesn't scale across subjects. This project automates that process: it ingests past exam papers (and optional syllabus/lecture material) for **any subject**, discovers the underlying topic structure directly from the content, and uses a language model — grounded in retrieved, relevant context — to draft new questions that plausibly reflect the observed exam pattern.

The system is intentionally domain-agnostic: no subject list, question bank, or topic taxonomy is hard-coded. Everything is derived from what the user uploads.

---

## 2. AI Techniques Implemented

This project satisfies the assignment's "minimum three techniques" requirement with the following:

| # | Technique | Where it's used |
|---|---|---|
| 1 | **NLP — text pre/post-processing** | `src/preprocessing/`, `src/classification/topic_classifier.py`: OCR text cleaning, tokenization, stopword removal, lemmatization, question segmentation |
| 2 | **Transformer-based models** | `src/embeddings/embedder.py` (BGE sentence embeddings), `src/retrieval/reranker.py` (cross-encoder reranking), `src/classification/bert_classifier.py` (zero-shot DistilBERT classification) |
| 3 | **Prompt engineering** | `src/generation/question_generator.py`: three selectable strategies — direct, chain-of-thought, and context-aware (retrieval-grounded) prompting for the Mistral LLM |
| 4 | **Retrieval-Augmented Generation** | `src/retrieval/pinecone_store.py` + reranker feed retrieved context into generation, evaluated quantitatively (see §6) |

---

## 3. System Architecture

```
                  ┌─────────────────────┐
 PDF Upload ───▶ │   Mistral OCR /     │
 (past papers,   │   pdfplumber/pypdf   │
 lecture notes)  │   extraction         │
                 └──────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │  Text Cleaning &    │
                 │  Question Segment.  │  (src/preprocessing)
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │  Sentence Embeddings│  (BAAI/bge-large-en-v1.5)
                 └──────────┬──────────┘
                            ▼
        ┌───────────────────┴────────────────────┐
        ▼                                        ▼
┌───────────────────┐                   ┌───────────────────────┐
│ KMeans Topic      │                   │ Pinecone Vector Index │
│ Clustering        │                   │ (semantic retrieval)  │
│ (dynamic, no fixed│                   └───────────┬───────────┘
│ topic list)       │                               ▼
└──────────┬────────┘                   ┌───────────────────────┐
           │                            │ Cross-Encoder Reranker│
           │                            └───────────┬───────────┘
           ▼                                        ▼
┌────────────────────────────────────────────────────────────────┐
│           Multi-Agent Workflow (LangGraph-style)               │
│ PastPaperAgent → LecturePdfAgent → PredictionAgent → Eval Agent│
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
                    ┌────────────────────────┐
                    │ Mistral LLM            │
                    │ (question generation,  │
                    │  prompt-engineered)    │
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ Streamlit Dashboard    │
                    │ (topics, predictions,  │
                    │  evaluation, export)   │
                    └────────────────────────┘
```

**Agent roles** (`src/agents/exam_agents.py`):
- **PastPaperAgent** — ingests past exam PDFs into structured question rows
- **LecturePdfAgent** — ingests syllabus/lecture PDFs into reference chunks
- **ExamAgentSystem** — indexes both corpora into Pinecone
- **PredictionAgent** — retrieves relevant context, reranks it, and prepares the generation prompt
- **EvaluationAgent** — reuses the retrieval path to compute ranking quality metrics

If `langgraph` is installed, these steps run as nodes in a `StateGraph`; otherwise the app falls back to the same steps called directly in Python, so the workflow behaves identically either way.

---

## 4. Features

- **Past exam paper upload** — extracts individual questions from PDF exam papers via Mistral OCR
- **Subject reference upload** — syllabus/notes/textbook PDFs indexed as retrieval context
- **Dynamic topic discovery** — KMeans clustering (silhouette-selected *k*) over embeddings, no predefined topics
- **Retrieval-augmented question prediction** — Pinecone retrieval + cross-encoder reranking + Mistral generation
- **Configurable prompting** — direct / chain-of-thought / context-aware strategies, adjustable difficulty
- **Multi-subject support** — any number of subjects, filterable throughout the dashboard
- **Retrieval evaluation** — Precision@k, Recall@k, MRR, nDCG computed on demand
- **Analytics dashboard** — topic trends, year-wise distribution, similarity search, CSV/PDF export

---

## 5. Tech Stack

| Layer | Tools |
|---|---|
| Interface | Streamlit, Plotly |
| OCR / extraction | Mistral OCR, pdfplumber, pypdf |
| NLP | NLTK, spaCy |
| Embeddings | Sentence-Transformers (`BAAI/bge-large-en-v1.5`) |
| Classification | DistilBERT zero-shot (`typeform/distilbert-base-uncased-mnli`) |
| Clustering | scikit-learn (KMeans, TF-IDF, silhouette score) |
| Vector store | Pinecone |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Generation | Mistral chat completions |
| Orchestration | LangGraph-style multi-agent workflow |
| Language / runtime | Python 3.12+ |

---

## 6. Retrieval Evaluation Results

Retrieval quality was measured against relevance-labelled queries over the indexed exam corpus (Precision/Recall/MRR/nDCG @ k = 5):

| Metric | Score | Rating |
|---|---|---|
| Precision@5 | 0.664 | Good |
| Recall@5 | 1.000 | Perfect |
| MRR | 0.513 | Moderate |
| nDCG@5 | 0.731 | Good |

**Interpretation:** Recall is perfect — no relevant past question is ever missed by the retriever, which matters most for this use case since a missed pattern means a missed prediction signal. MRR indicates the single best match typically lands around rank 2 rather than rank 1, suggesting the cross-encoder reranker has room to improve with domain-specific fine-tuning. Full methodology, formulas, and improvement suggestions are in the [Retrieval Evaluation Metrics](#retrieval-evaluation-metrics-detail) section below and can be reproduced live from the app's **Retrieval Evaluation** tab.

---

## 7. Project Structure

```
Exam-Pattern-Analysis/
├── app/
│   └── streamlit_app.py         # Main dashboard entry point
├── src/
│   ├── preprocessing/            # PDF extraction, OCR, text cleaning
│   ├── ocr/                      # Mistral OCR client
│   ├── embeddings/               # Sentence embedding + caching
│   ├── classification/           # Topic clustering, BERT question typing
│   ├── retrieval/                # Pinecone store, cross-encoder reranker
│   ├── generation/                # Prompt-engineered question generation
│   ├── evaluation/                # Retrieval metrics, analytics
│   ├── agents/                   # Multi-agent orchestration (LangGraph)
│   ├── pipeline.py               # Data loading + analysis pipeline
│   └── utils.py                  # Shared paths/logging
├── scripts/                       # Setup, validation, smoke-test utilities
├── data/
│   ├── raw/                       # Uploaded source PDFs
│   └── processed/                 # Generated questions.csv, materials.csv
├── reports/                       # Final report, organized outputs, logs
├── notebooks/                     # Exploratory notebooks
└── requirements.txt
```

---

## 8. Setup & Installation

```powershell
git clone https://github.com/Parakkrama24/Exam-Pattern-Analysis.git
cd Exam-Pattern-Analysis
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Create a `.env` file in the project root with:

```env
MISTRAL_API_KEY=your-real-key-here
PINECONE_API_KEY=your-real-key-here
PINECONE_INDEX_NAME=quickstart
MISTRAL_CHAT_MODEL=mistral-medium-latest
```

## 9. Running the App

```powershell
streamlit run app\streamlit_app.py
```

## 10. How to Use

1. **Upload past exam papers** — *Upload & Process → Past Exam Papers*: enter a subject name and year, upload PDFs, click **Process**.
2. **Upload subject reference material** *(optional, recommended)* — *Upload & Process → Subject PDFs*: syllabus/notes/textbooks, OCR-indexed as retrieval context.
3. **Explore results**:
   - **Topic Analysis** — discovered topics, filterable by subject
   - **Question Predictions** — pick a subject + topic, choose a prompting strategy and difficulty, generate new questions
   - **Similarity Search** — semantic search across the indexed corpus
   - **Retrieval Evaluation** — live Precision@k / Recall@k / MRR / nDCG
   - **Analytics Dashboard** — corpus-wide statistics and trends

**PDF tips:** use text-based PDFs where possible; numbered questions (`1.`, `Q1.`) parse most reliably; image-only PDFs are still supported via OCR but will be slower.

---

## Retrieval Evaluation Metrics (detail)

<details>
<summary>Expand for metric formulas, score interpretation bands, and improvement strategies</summary>

#### Precision@5
`Precision@5 = (relevant results in top 5) / 5`
Our result: **0.664** — roughly 3–4 of 5 retrieved chunks are relevant; Mistral's instruction-following tolerates the remaining noise reasonably well.

#### Recall@5
`Recall@5 = (relevant results found in top 5) / (total relevant results in corpus)`
Our result: **1.000** — every relevant past question/chunk is retrieved. This is the highest-priority metric for the project, since a missed relevant question means a pattern never reaches the generator.

#### MRR (Mean Reciprocal Rank)
`MRR = average(1 / rank of first relevant result)`
Our result: **0.513** — the first relevant result appears around rank 2 on average; a stronger reranker signal would push this toward 1.0.

#### nDCG@5
`nDCG@5 = DCG@5 / Ideal DCG@5`
Our result: **0.731** — relevant content is generally ranked near the top, which matters because Mistral processes retrieved chunks in order.

**Improvement strategies:** stricter similarity thresholds before reranking (Precision); domain-specific reranker fine-tuning (MRR); wider candidate pool before reranking, e.g. top-20 → top-5 (nDCG); current embedding/indexing already achieves ceiling performance on Recall.

</details>

---

## Known Limitations

- Requires valid `MISTRAL_API_KEY` and `PINECONE_API_KEY` to run end-to-end; without them, OCR, retrieval, and generation are unavailable.
- Topic labels are derived from TF-IDF keywords over clusters, so label quality depends on corpus size — very small uploads (a handful of questions) produce less meaningful clusters.
- Scanned/image-only PDFs rely on Mistral OCR and take longer to process than text-based PDFs.

