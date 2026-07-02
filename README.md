# AI-Powered Exam Pattern Analysis and Question Prediction System

Analyze **your own** past examination papers using Mistral OCR, Pinecone vector retrieval, reranking, and multi-agent orchestration to discover question patterns, identify important topics, and generate probable future exam questions.

> **No sample CSV data.** All analysis comes from PDFs you upload. Subjects are fully dynamic — enter any subject name (e.g. Physics, Law, Nursing, Finance).

## Features

- **Past Exam Paper Upload** — Extract questions from PDF exam papers with Mistral OCR
- **Subject PDF Upload** — Add syllabus, notes, or textbook PDFs as Mistral OCR context
- **Dynamic Topic Discovery** — KMeans clustering on your exam content (no predefined topics)
- **Retrieval-Augmented Prediction** — Pinecone stores embeddings and reranking improves retrieval quality
- **Multi-Agent Layout** — Separate past-paper, lecture-pdf, prediction, and evaluation agents
- **Mistral Question Generation** — Mistral chat completions generate new questions from your patterns
- **Multi-Subject Support** — Upload papers for multiple subjects and filter analysis
- **Interactive Dashboard** — Topic charts, similarity search, retrieval evaluation, analytics, CSV/PDF export

## Tech Stack

- Python 3.12+, Streamlit, Mistral API, Pinecone
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
6. The prediction agent sends the reranked context into Mistral chat completions.
7. Evaluation metrics measure retrieval quality from the indexed corpus.

##  API Key

Required for **Question Predictions**. Provide via:
- `API_KEY` in `.env`

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
├── src/                   # NLP, clustering, OpenAI generation
└── scripts/               # Utility scripts
```

## Screenshots


## Team Members


## License

MIT License
