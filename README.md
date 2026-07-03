# AI-Powered Exam Pattern Analysis and Question Prediction System

Analyze **your own** past examination papers using Mistral OCR, Pinecone vector retrieval, reranking, and a LangGraph-style multi-agent workflow to discover question patterns, identify important topics, and generate probable future exam questions.

> **No sample CSV data.** All analysis comes from PDFs you upload. Subjects are fully dynamic — enter any subject name (e.g. Physics, Law, Nursing, Finance).

## Features

- **Past Exam Paper Upload** — Extract questions from PDF exam papers with Mistral OCR
- **Subject PDF Upload** — Add syllabus, notes, or textbook PDFs as Mistral OCR context
- **Dynamic Topic Discovery** — KMeans clustering on your exam content (no predefined topics)
- **Retrieval-Augmented Prediction** — Pinecone stores embeddings and reranking improves retrieval quality
- **LangGraph-Style Workflow** — Separate past-paper, lecture-pdf, prediction, and evaluation agents coordinated in one graph
- **Mistral Question Generation** — Mistral chat completions generate new questions from your patterns
- **Multi-Subject Support** — Upload papers for multiple subjects and filter analysis
- **Interactive Dashboard** — Topic charts, similarity search, retrieval evaluation, analytics, CSV/PDF export

## Tech Stack

- Python 3.12+, Streamlit, LangGraph, Mistral API, Pinecone
- Sentence Transformers, Scikit-learn, NLTK
- Plotly, Pandas, Mistral OCR, cross-encoder reranking

## Setup

```powershell
Set-Location C:\Users\User\exam-pattern-analysis
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
copy .env.example .env
```

Edit `.env` and set your API keys:

```env
MISTRAL_API_KEY=your-real-key-here
PINECONE_API_KEY=your-real-key-here
PINECONE_INDEX_NAME=quickstart
MISTRAL_CHAT_MODEL=mistral-medium-latest
```

## Run

```powershell
streamlit run app\streamlit_app.py
```

## How to Use

### 1. Upload Past Exam Papers
- Go to **Upload & Process → Past Exam Papers**
- Enter any **subject name** (not limited to a fixed list)
- Set the **exam year**
- Upload one or more PDF exam papers
- Click **Process Exam Papers**

### 2. Upload Subject Reference PDFs (optional but recommended)
- Go to **Upload & Process → Subject PDFs**
- Enter the matching **subject name**
- Upload syllabus, notes, or textbook PDFs
- These are OCR-processed by Mistral and indexed in Pinecone as retrieval context

### 3. Analyze & Generate
- **Topic Analysis** — Discovered topics from your papers (filter by subject in sidebar)
- **Question Predictions** — Select subject + discovered topic → retrieval is reranked and Mistral generates new questions
- **Similarity Search** — Pinecone-backed semantic search over uploaded question bank and lecture chunks
- **Retrieval Evaluation** — Precision@k, Recall@k, MRR, and nDCG for retrieval quality
- **Analytics Dashboard** — Stats across all uploaded subjects

### Architecture

1. Upload a PDF in the Streamlit app.
2. Mistral OCR converts the PDF into page-level Markdown/text.
3. Past-paper PDFs become question rows; lecture PDFs become chunk rows.
4. Sentence embeddings are created and stored in Pinecone.
5. Retrieval pulls the most relevant chunks, then a cross-encoder reranks them.
6. The workflow routes into retrieval, reranking, and Mistral chat completions.
7. Evaluation metrics measure retrieval quality from the indexed corpus.

##  API Key

Required runtime keys:
- `MISTRAL_API_KEY` for OCR and question generation
- `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` for retrieval

## PDF Tips

- Use **text-based PDFs** (not scanned images)
- Exam papers should have **numbered questions** (`1.`, `Q1.`, etc.)
- Image-based PDFs show a friendly error message

## Project Structure

```
exam-pattern-analysis/
├── data/raw/              # Uploaded PDFs saved here
├── data/processed/        # questions.csv, subject_materials.csv
├── app/streamlit_app.py   # Main dashboard
├── src/                   # NLP, clustering, retrieval, generation, workflow agents
└── scripts/               # Utility scripts
```

## Retrieval Evaluation Metrics

The **Retrieval Evaluation** tab in the dashboard measures how well the Pinecone vector retrieval + cross-encoder reranking pipeline performs. These metrics are computed against a set of relevance-labelled queries over the indexed exam question corpus.

### What Is Being Evaluated?

When you select a topic and generate questions, the system:
1. Embeds your query using Sentence Transformers
2. Retrieves the top-K most similar chunks from Pinecone
3. Reranks them using a cross-encoder model
4. Feeds the reranked context into Mistral for question generation

### Agent Workflow

The app keeps the agent logic in a single Python process, but the control flow is organized like a LangGraph pipeline:

1. `PastPaperAgent` ingests past exam PDFs into question rows.
2. `LecturePdfAgent` ingests syllabus/notes PDFs into reference chunks.
3. `ExamAgentSystem` indexes both corpora into Pinecone.
4. `PredictionAgent` retrieves context, reranks results, and prepares generation inputs.
5. `EvaluationAgent` reuses the retrieval path to compute ranking metrics.

If `langgraph` is installed, the workflow wrapper in `src/agents/exam_agents.py` routes these steps through a graph object; otherwise the app falls back to the same imperative Python calls.

The evaluation metrics measure the **quality of steps 2–3** — how relevant and well-ordered the retrieved results are.

---

### Metrics Explained

#### 🎯 Precision@5
> *Out of the 5 results returned, how many are actually relevant?*

**Formula:**
```
Precision@5 = (Number of relevant results in top 5) / 5
```

| Score Range | Meaning |
|-------------|---------|
| 1.00 | All 5 results are relevant — perfect |
| 0.60–0.79 | 3–4 relevant results — good |
| < 0.40 | More than half are irrelevant — poor |

**Our Result: `0.664`** — About 3–4 out of 5 retrieved chunks are relevant. Acceptable, as Mistral can filter minor noise from the context.

**Relevance to this project:** Medium — even if 1–2 chunks are off-topic, Mistral's instruction following handles the noise reasonably well.

---

#### 🔍 Recall@5
> *Out of ALL relevant documents in the corpus, how many did the system find within the top 5?*

**Formula:**
```
Recall@5 = (Relevant results found in top 5) / (Total relevant results in corpus)
```

| Score Range | Meaning |
|-------------|---------|
| 1.00 | Every relevant result was retrieved — perfect |
| 0.70–0.99 | Most relevant content retrieved — good |
| < 0.50 | Significant relevant content missed — poor |

**Our Result: `1.000`** ⭐ — Every relevant past exam question/chunk is retrieved within the top 5. Nothing is missed.

**Relevance to this project:** **Highest priority.** Missing a relevant past question means a key exam pattern is never passed to Mistral, leading to weaker predictions. A perfect score here is critical.

---

#### 🥇 MRR (Mean Reciprocal Rank)
> *On average, at what position does the FIRST relevant result appear?*

**Formula:**
```
MRR = average of (1 / rank of first relevant result) across all queries
```

| Score | First Relevant Result Appears At |
|-------|----------------------------------|
| 1.00 | Position 1 (always first) |
| 0.50 | Position ~2 |
| 0.33 | Position ~3 |
| < 0.25 | Position 4 or later — poor |

**Our Result: `0.513`** — The most relevant result appears at approximately **rank 2** on average.

**Relevance to this project:** High — the cross-encoder reranker should push the best-matching past question to rank 1. An MRR below 0.6 suggests the reranker ordering could be improved.

---

#### 📈 nDCG@5 (Normalized Discounted Cumulative Gain)
> *Are the most relevant results ranked highest? Results at lower positions are penalised.*

**Formula:**
```
nDCG@5 = DCG@5 / Ideal DCG@5

DCG@5 = sum of (relevance_score / log2(rank + 1)) for rank 1 to 5
```

| Score Range | Meaning |
|-------------|---------|
| 0.90–1.00 | Near-perfect ranking — excellent |
| 0.70–0.89 | Good ranking quality |
| 0.50–0.69 | Moderate — ranking could be improved |
| < 0.50 | Poor ranking — relevant results buried |

**Our Result: `0.731`** — Relevant content is ranked near the top. The reranker is performing well overall.

**Relevance to this project:** Very high — Mistral processes retrieved chunks in order. If the most relevant content is buried at rank 5, the generated questions will be less accurate. Good nDCG means better question generation quality.

---

### Results Summary

| Metric | Score | Rating | Priority for This Project |
|--------|-------|--------|--------------------------|
| **Precision@5** | 0.664 | ✅ Good | Medium |
| **Recall@5** | 1.000 | 🏆 Perfect | **Highest** |
| **MRR** | 0.513 | ⚠️ Moderate | High |
| **nDCG@5** | 0.731 | ✅ Good | Very High |

---

### How to Improve Scores

| Metric | Improvement Strategy |
|--------|---------------------|
| **Precision@5** | Use stricter similarity thresholds in Pinecone; filter low-score chunks before reranking |
| **MRR** | Fine-tune the cross-encoder reranker on domain-specific (exam) data |
| **nDCG@5** | Increase the candidate pool (retrieve top-20, rerank to top-5) for better reranking signal |
| **Recall@5** | Already perfect — maintain current embedding model and indexing strategy |

---

### Where to View Metrics in the App

Navigate to **Retrieval Evaluation** tab in the Streamlit dashboard to:
- Run live evaluation against your uploaded exam corpus
- Adjust the value of `k` (default: 5) to test Precision@k, Recall@k, and nDCG@k
- Compare performance before and after uploading additional reference PDFs

---

## Screenshots


## Team Members


## License

MIT License
