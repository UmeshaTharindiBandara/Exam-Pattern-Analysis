# AI-Powered Exam Pattern Analysis and Question Prediction System

Analyze **your own** past examination papers using NLP and OpenAI to discover question patterns, identify important topics, and generate probable future exam questions.

> **No sample CSV data.** All analysis comes from PDFs you upload. Subjects are fully dynamic — enter any subject name (e.g. Physics, Law, Nursing, Finance).

## Features

- **Past Exam Paper Upload** — Extract questions from PDF exam papers (any subject)
- **Subject PDF Upload** — Add syllabus, notes, or textbook PDFs as OpenAI context
- **Dynamic Topic Discovery** — KMeans clustering on your exam content (no predefined topics)
- **OpenAI Question Generation** — GPT-4o generates new questions from your patterns
- **Multi-Subject Support** — Upload papers for multiple subjects and filter analysis
- **Interactive Dashboard** — Topic charts, similarity search, analytics, CSV/PDF export

## Tech Stack

- Python 3.12+, Streamlit, OpenAI API
- Sentence Transformers, Scikit-learn, NLTK
- pdfplumber, Plotly, Pandas

## Setup

```powershell
Set-Location C:\Users\User\exam-pattern-analysis
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
copy .env.example .env
```

Edit `.env` and set your API key:

```env
GEMINI_API_KEY=AQ.your-real-key-here
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
- These are sent to OpenAI as context when generating questions

### 3. Analyze & Generate
- **Topic Analysis** — Discovered topics from your papers (filter by subject in sidebar)
- **Question Predictions** — Select subject + discovered topic → OpenAI generates new questions
- **Similarity Search** — Search your uploaded question bank
- **Analytics Dashboard** — Stats across all uploaded subjects

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
