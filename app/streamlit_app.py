"""Streamlit dashboard for AI-powered exam pattern analysis."""

from __future__ import annotations

import os
import re
import tempfile
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from wordcloud import WordCloud

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.exam_agents import ExamAgentSystem
from src.embeddings.embedder import QuestionEmbedder
from src.evaluation.evaluator import ExamEvaluator
from src.pipeline import (
    NoQuestionDataError,
    filter_by_subject,
    get_subject_context,
    get_subjects,
    run_analysis_pipeline,
)
from src.preprocessing.pdf_extractor import PDFExtractor
from src.utils import setup_logging

logger = setup_logging("streamlit_app")

st.set_page_config(
    page_title="Exam Pattern Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_CSS = """
<style>
    /* Question cards — readable in both light and dark Streamlit themes */
    .question-card {
        background-color: #f0f4ff;
        border: 1px solid #c5d5f5;
        border-left: 4px solid #4f8ef7;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        color: #1a2240;
    }

    /* Dark theme override (Streamlit sets data-theme on html element) */
    html[data-theme="dark"] .question-card,
    [data-testid="stAppViewContainer"][class*="dark"] .question-card {
        background-color: #1c2645;
        border-color: #2e4a8a;
        border-left-color: #5b9cf6;
        color: #d8e4ff;
    }

    /* MCQ options list inside question cards */
    .question-card .mcq-question {
        margin-bottom: 10px;
        font-size: 0.97rem;
        line-height: 1.55;
    }
    .question-card .mcq-options {
        list-style: none;
        padding: 0;
        margin: 8px 0 0 0;
    }
    .question-card .mcq-options li {
        padding: 5px 10px;
        margin: 4px 0;
        border-radius: 6px;
        background-color: rgba(79, 142, 247, 0.08);
        border: 1px solid rgba(79, 142, 247, 0.18);
        font-size: 0.93rem;
    }
    html[data-theme="dark"] .question-card .mcq-options li {
        background-color: rgba(91, 156, 246, 0.12);
        border-color: rgba(91, 156, 246, 0.25);
    }

    /* Metric cards: subtle tinted background for separation */
    [data-testid="metric-container"] {
        background-color: rgba(79, 142, 247, 0.07);
        border: 1px solid rgba(79, 142, 247, 0.18);
        border-radius: 10px;
        padding: 14px 18px;
    }

    /* Sidebar separator */
    div[data-testid="stSidebar"] > div:first-child {
        border-right: 1px solid rgba(100, 116, 139, 0.25);
    }

    /* Improve tab label contrast */
    button[data-baseweb="tab"] {
        font-weight: 500;
    }

    /* Rounded, clearly visible primary buttons */
    .stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Secondary buttons: clearly visible border */
    .stButton > button[kind="secondary"] {
        border-radius: 8px;
        border-width: 1.5px;
    }

    /* Dataframe header row contrast boost */
    [data-testid="stDataFrame"] th {
        font-weight: 600 !important;
    }

    /* Cross-paper duplicate pair cards */
    .dup-card {
        border: 1px solid #c5d5f5;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 14px;
        background-color: #f7f9ff;
    }
    html[data-theme="dark"] .dup-card,
    [data-testid="stAppViewContainer"][class*="dark"] .dup-card {
        background-color: #1a2340;
        border-color: #2e4a8a;
    }
    .dup-score {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 10px;
    }
    .dup-score.exact  { color: #d62728; }
    .dup-score.high   { color: #e07b00; }
    .dup-score.medium { color: #1f77b4; }
    .dup-paper-label {
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 4px;
        color: #4f8ef7;
    }
    .dup-q {
        font-size: 0.93rem;
        line-height: 1.5;
    }
    .dup-divider {
        border: none;
        border-top: 1px dashed rgba(100,116,139,0.3);
        margin: 6px 0 10px 0;
    }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

import re as _re

_MCQ_OPTION_RE = _re.compile(
    r"(?<![\w])([A-D])[.):]\s+",
    _re.IGNORECASE,
)


def _split_mcq_options(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a question string into the stem and MCQ options if present.

    Returns:
        (stem, options_list) where options_list is a list of (letter, text) tuples.
        If no options are detected both returned values represent the full text.
    """
    # Find positions of A. / B. / C. / D. option markers
    positions = [(m.group(1).upper(), m.start()) for m in _MCQ_OPTION_RE.finditer(text)]
    # Only treat as MCQ when at least 2 sequential options are found starting with A
    letters = [p[0] for p in positions]
    if len(positions) < 2 or "A" not in letters:
        return text, []

    first_option_start = positions[0][1]
    stem = text[:first_option_start].strip()
    options: list[tuple[str, str]] = []
    for i, (letter, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        # Strip the leading "A. " / "B. " marker itself before recording option text
        option_text = _MCQ_OPTION_RE.sub("", text[start:end], count=1).strip()
        options.append((letter, option_text))
    return stem, options


def render_question_card(idx: int, item: dict) -> str:
    """Build an HTML question card with MCQ options on separate lines."""
    label = f"Q{idx}. [{item.get('type', 'N/A').upper()} | {item.get('marks', 0)} marks]"
    question_raw = item.get("question", "")
    stem, options = _split_mcq_options(question_raw)

    options_html = ""
    if options:
        li_items = "".join(
            f"<li><strong>{letter}.</strong>&nbsp;{opt_text}</li>"
            for letter, opt_text in options
        )
        options_html = f'<ul class="mcq-options">{li_items}</ul>'

    return (
        f'<div class="question-card">'
        f"<strong>{label}</strong><br/>"
        f'<div class="mcq-question">{stem}</div>'
        f"{options_html}"
        f"</div>"
    )


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "questions_raw_df": None,
        "subject_materials_raw_df": None,
        "questions_df": None,
        "topics_df": None,
        "embeddings": None,
        "subject_materials_df": None,
        "generated_questions": [],
        "analysis_ready": False,
        "prompt_strategy": "context_aware",
        "api_key": "",
        "selected_subject_filter": "All Subjects",
        "index_sync_counts": {"questions": 0, "materials": 0},
        "retrieval_metrics": None,
        "retrieval_details": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_full_analysis(force: bool = False) -> bool:
    """Load uploaded PDF data and run the NLP analysis pipeline.

    Args:
        force: Force pipeline rerun.

    Returns:
        True if analysis completed successfully.
    """
    if st.session_state.analysis_ready and not force:
        return st.session_state.questions_df is not None

    raw_questions_df = st.session_state.questions_raw_df
    if raw_questions_df is None or raw_questions_df.empty:
        st.info("Upload past exam PDFs to begin analysis.")
        st.session_state.analysis_ready = False
        return False

    try:
        cache_key = f"exam_questions_{len(raw_questions_df)}"
        annotated, topics, embeddings = run_analysis_pipeline(
            raw_questions_df,
            force_recompute=force,
            cache_key=cache_key,
            embedder=get_embedder(),
        )
        st.session_state.questions_df = annotated
        st.session_state.topics_df = topics
        st.session_state.embeddings = embeddings
        st.session_state.subject_materials_df = st.session_state.subject_materials_raw_df
        agent_system = get_agent_system()
        st.session_state.index_sync_counts = agent_system.run_index_corpus(
            annotated,
            st.session_state.subject_materials_df,
            question_embeddings=embeddings,
        )
        st.session_state.analysis_ready = True
        return True
    except NoQuestionDataError as exc:
        st.session_state.analysis_ready = False
        st.session_state.questions_df = None
        st.info(str(exc))
        return False
    except Exception as exc:
        logger.exception("Analysis pipeline failed: %s", exc)
        st.error(f"Analysis failed: {exc}")
        return False


def get_filtered_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Return questions, topics, and embeddings filtered by selected subject."""
    questions_df = st.session_state.questions_df
    topics_df = st.session_state.topics_df
    embeddings = st.session_state.embeddings
    subject_filter = st.session_state.selected_subject_filter

    if subject_filter == "All Subjects":
        return questions_df, topics_df, embeddings

    filtered_q = filter_by_subject(questions_df, subject_filter)
    if filtered_q.empty:
        return filtered_q, topics_df.iloc[0:0], None

    topic_labels = filtered_q["topic_label"].unique().tolist()
    filtered_t = topics_df[topics_df["topic_label"].isin(topic_labels)].copy()
    indices = filtered_q.index.to_numpy()
    filtered_e = embeddings[indices] if embeddings is not None else None
    return filtered_q, filtered_t, filtered_e


def subject_filter_widget() -> None:
    """Render subject filter dropdown in the sidebar."""
    subjects: list[str] = []
    if st.session_state.questions_df is not None:
        subjects = get_subjects(st.session_state.questions_df)

    options = ["All Subjects"] + subjects
    current = st.session_state.selected_subject_filter
    if current not in options:
        current = "All Subjects"

    st.session_state.selected_subject_filter = st.sidebar.selectbox(
        "Filter by Subject",
        options,
        index=options.index(current),
    )


def render_sidebar() -> None:
    """Render sidebar navigation and settings."""
    st.sidebar.title("Exam Pattern AI")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Upload & Process",
            "Topic Analysis",
            "Question Predictions",
            "Similarity Search",
            "Retrieval Evaluation",
            "Analytics Dashboard",
            "Evaluation Metrics",
        ],
    )

    st.sidebar.divider()

    subject_filter_widget()

    if st.sidebar.button("Refresh Analysis"):
        st.session_state.analysis_ready = False
        run_full_analysis(force=True)
        st.sidebar.success("Analysis refreshed.")

    if st.sidebar.button("Clear All Uploaded Data", type="secondary"):
        agent_system = get_agent_system()
        agent_system.vector_store.delete_namespace("past-papers")
        agent_system.vector_store.delete_namespace("lecture-pdfs")
        st.session_state.analysis_ready = False
        st.session_state.questions_raw_df = None
        st.session_state.subject_materials_raw_df = None
        st.session_state.questions_df = None
        st.session_state.topics_df = None
        st.session_state.embeddings = None
        st.session_state.subject_materials_df = None
        st.session_state.generated_questions = []
        st.sidebar.warning("All uploaded data cleared.")

    st.session_state.current_page = page

    st.sidebar.divider()


@st.cache_resource
def get_embedder() -> QuestionEmbedder:
    """Return the single process-wide embedding model instance.

    Streamlit reruns the whole script on every widget interaction, and each
    browser tab gets its own session_state. Without this cache, every session
    would load its own ~1.3GB copy of the embedding model concurrently, which
    is enough concurrent native memory pressure to crash the process.
    """
    return QuestionEmbedder()


@st.cache_resource
def get_agent_system() -> ExamAgentSystem:
    """Return the shared multi-agent orchestration system."""
    return ExamAgentSystem(embedder=get_embedder())


def page_upload_process() -> None:
    """Render upload and PDF processing page."""
    st.title("Upload & Process")
    st.write(
        "Upload **past exam papers** and **subject reference PDFs** (syllabus, notes, textbooks). "
        "Enter any subject name — there are no predefined subjects."
    )

    tab_exam, tab_subject, tab_data = st.tabs(
        ["Past Exam Papers", "Subject PDFs", "Uploaded Data"]
    )

    with tab_exam:
        st.subheader("Upload Past Exam Papers")
        exam_subject = st.text_input(
            "Subject name for these exam papers",
            placeholder="e.g. Organic Chemistry, Data Structures, Constitutional Law",
            key="exam_subject_input",
        )
        exam_year = st.number_input(
            "Exam year",
            min_value=1990,
            max_value=2100,
            value=2024,
            key="exam_year_input",
        )
        exam_files = st.file_uploader(
            "Select exam paper PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="exam_uploader",
        )

        if st.button("Process Exam Papers", type="primary"):
            if not exam_subject.strip():
                st.error("Please enter a subject name before uploading.")
            elif not exam_files:
                st.error("Please select at least one exam paper PDF.")
            else:
                extractor = PDFExtractor()
                progress = st.progress(0)
                processed_frames: list[pd.DataFrame] = []
                errors: list[str] = []
                ocr_files: list[str] = []

                for idx, uploaded in enumerate(exam_files):
                    temp_path: Path | None = None
                    try:
                        safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", uploaded.name).strip(" ._") or "uploaded.pdf"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{safe_name}") as temp_file:
                            temp_file.write(uploaded.getvalue())
                            temp_path = Path(temp_file.name)
                        frame = extractor.process_pdf(
                            temp_path,
                            subject=exam_subject.strip(),
                            year=int(exam_year),
                        )
                        processed_frames.append(frame)
                    except Exception as exc:
                        logger.exception("PDF processing error: %s", exc)
                        errors.append(f"{uploaded.name}: {exc}")
                    finally:
                        if temp_path is not None and temp_path.exists():
                            try:
                                temp_path.unlink()
                            except Exception:
                                pass

                    progress.progress((idx + 1) / len(exam_files))

                if processed_frames:
                    combined = pd.concat(processed_frames, ignore_index=True)
                    existing = st.session_state.questions_raw_df
                    if existing is not None and not existing.empty:
                        combined = pd.concat([existing, combined], ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=["question_text", "year", "subject"],
                        keep="last",
                    ).reset_index(drop=True)

                    st.session_state.questions_raw_df = combined
                    st.session_state.analysis_ready = False
                    run_full_analysis(force=True)
                    st.success(
                        f"Extracted {len(combined)} questions for **{exam_subject.strip()}**. "
                        f"Total questions in Pinecone/session: {len(combined)}."
                    )
                    if ocr_files:
                        st.info(
                            f"**OCR was used** for {len(ocr_files)} scanned PDF(s): "
                            + ", ".join(ocr_files)
                        )

                for err in errors:
                    st.error(err)

    with tab_subject:
        st.subheader("Upload Subject Reference PDFs")
        st.caption(
            "Syllabus, textbook chapters, or lecture notes. Used as context when generating "
            "new questions via Gemini."
        )
        ref_subject = st.text_input(
            "Subject name for these reference PDFs",
            placeholder="e.g. Organic Chemistry, Machine Learning, History",
            key="ref_subject_input",
        )
        ref_files = st.file_uploader(
            "Select subject/syllabus PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="ref_uploader",
        )

        if st.button("Process Subject PDFs"):
            if not ref_subject.strip():
                st.error("Please enter a subject name before uploading.")
            elif not ref_files:
                st.error("Please select at least one subject PDF.")
            else:
                extractor = PDFExtractor()
                progress = st.progress(0)
                material_frames: list[pd.DataFrame] = []
                errors: list[str] = []
                ocr_ref_files: list[str] = []

                for idx, uploaded in enumerate(ref_files):
                    temp_path: Path | None = None
                    try:
                        safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", uploaded.name).strip(" ._") or "uploaded.pdf"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{safe_name}") as temp_file:
                            temp_file.write(uploaded.getvalue())
                            temp_path = Path(temp_file.name)
                        frame = extractor.process_subject_pdf(
                            temp_path, subject=ref_subject.strip()
                        )
                        material_frames.append(frame)
                    except Exception as exc:
                        logger.exception("Subject PDF error: %s", exc)
                        errors.append(f"{uploaded.name}: {exc}")
                    finally:
                        if temp_path is not None and temp_path.exists():
                            try:
                                temp_path.unlink()
                            except Exception:
                                pass

                    progress.progress((idx + 1) / len(ref_files))

                if material_frames:
                    combined = pd.concat(material_frames, ignore_index=True)
                    existing = st.session_state.subject_materials_raw_df
                    if existing is not None and not existing.empty:
                        combined = pd.concat([existing, combined], ignore_index=True)
                    if "chunk_id" not in combined.columns:
                        combined["chunk_id"] = combined["source_file"].astype(str)
                    combined = combined.drop_duplicates(
                        subset=["subject", "source_file", "chunk_id"],
                        keep="last",
                    ).reset_index(drop=True)

                    st.session_state.subject_materials_raw_df = combined
                    st.session_state.subject_materials_df = combined
                    st.session_state.analysis_ready = False
                    run_full_analysis(force=True)
                    st.success(
                        f"Saved {len(combined)} subject reference document(s) for "
                        f"**{ref_subject.strip()}**."
                    )
                    if ocr_ref_files:
                        st.info(
                            f"**OCR was used** for {len(ocr_ref_files)} scanned PDF(s): "
                            + ", ".join(ocr_ref_files)
                        )

                for err in errors:
                    st.error(err)

    with tab_data:
        if run_full_analysis():
            df = st.session_state.questions_df
            st.metric("Total Questions", len(df))
            st.metric("Subjects", len(get_subjects(df)))
            st.metric("Source PDFs", df["source_file"].nunique() if "source_file" in df else 0)

            materials = st.session_state.subject_materials_df
            if materials is not None and not materials.empty:
                st.subheader("Subject Reference PDFs")
                st.dataframe(
                    materials[["subject", "source_file"]],
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("Extracted Questions")
            st.dataframe(df, use_container_width=True, hide_index=True)


def page_topic_analysis() -> None:
    """Render topic analysis visualizations."""
    st.title("Topic Analysis")
    if not run_full_analysis():
        return

    questions_df, topics_df, _ = get_filtered_data()
    if questions_df.empty:
        st.warning("No questions for the selected subject filter.")
        return

    evaluator = ExamEvaluator()
    subject_note = (
        f" — {st.session_state.selected_subject_filter}"
        if st.session_state.selected_subject_filter != "All Subjects"
        else ""
    )

    top_topics = evaluator.top_topics_bar_data(topics_df, top_n=10)
    if not top_topics.empty:
        fig = evaluator.make_bar_chart(
            top_topics,
            x="question_count",
            y="topic_label",
            title=f"Top Topics Discovered from Your Exam Papers{subject_note}",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    trend_data = evaluator.topic_trend_line_data(topics_df)
    if not trend_data.empty:
        line_fig = evaluator.make_line_chart(
            trend_data,
            x="year",
            y="count",
            color="topic_label",
            title="Topic Frequency Trends Over Years",
        )
        st.plotly_chart(line_fig, use_container_width=True)

    if not topics_df.empty:
        st.subheader("Topic Details")
        selected_topic = st.selectbox("Select topic for word cloud", topics_df["topic_label"].tolist())
        topic_questions = questions_df[questions_df["topic_label"] == selected_topic]
        if not topic_questions.empty:
            text_blob = " ".join(topic_questions["cleaned_text"].fillna("").tolist())
            if text_blob.strip():
                wc = WordCloud(
                    width=900,
                    height=400,
                    background_color="#f0f4ff",
                    colormap="Blues",
                ).generate(text_blob)
                st.image(wc.to_array(), use_container_width=True)

        display_df = topics_df.copy()
        if "trend" not in display_df.columns:
            display_df["trend"] = "stable"
        display_df["sample_questions"] = display_df["sample_questions"].apply(
            lambda items: " | ".join(items[:2]) if isinstance(items, list) else str(items)
        )
        st.dataframe(
            display_df[["topic_label", "question_count", "trend", "sample_questions"]],
            use_container_width=True,
            hide_index=True,
        )


def page_question_predictions() -> None:
    """Render LLM question prediction page."""
    st.title("Question Predictions")
    st.caption("Requires a valid Mistral API key. Questions are generated from your uploaded exam patterns.")

    if not run_full_analysis():
        return

    questions_df, topics_df, _ = get_filtered_data()
    if questions_df.empty or topics_df.empty:
        st.warning("Upload exam papers and select a subject with data to generate questions.")
        return

    subjects_in_view = get_subjects(questions_df)
    gen_subject = st.selectbox(
        "Subject",
        subjects_in_view,
        help="Subject context sent to Gemini along with discovered topics.",
    )
    subject_questions = questions_df[questions_df["subject"] == gen_subject]
    subject_topics = topics_df[
        topics_df["topic_label"].isin(subject_questions["topic_label"].unique())
    ]

    if subject_topics.empty:
        st.warning("No discovered topics found for the selected subject yet.")
        return

    topic = st.selectbox("Discovered Topic", subject_topics["topic_label"].tolist())
    num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=5)
    difficulty = st.radio("Difficulty Level", ["Easy", "Medium", "Hard"], horizontal=True)

    if st.button("Generate Questions with Mistral", type="primary"):
        if not st.session_state.api_key and not _env_mistral_key():
            st.error("Mistral API key is required. Enter it in the sidebar or set MISTRAL_API_KEY in .env.")
        else:
            try:
                with st.spinner("Generating questions via Mistral..."):
                    agent_system = get_agent_system()
                    materials_df = (
                        st.session_state.subject_materials_df
                        if st.session_state.subject_materials_df is not None
                        else load_subject_materials()
                    )
                    generated = agent_system.generate_questions(
                        topic=topic,
                        subject=gen_subject,
                        num_questions=num_questions,
                        difficulty=difficulty,
                        strategy=st.session_state.prompt_strategy,
                        questions_df=questions_df,
                        topics_df=topics_df,
                        materials_df=materials_df,
                    )
                st.session_state.generated_questions = generated
                st.session_state.gen_topic_for_bleu = topic
                st.session_state.gen_subject_for_bleu = gen_subject
            except Exception as exc:
                logger.exception("Generation failed: %s", exc)
                st.error(f"Generation failed: {exc}")

    if st.session_state.generated_questions:
        st.subheader(f"Generated Questions — {gen_subject}")
        for idx, item in enumerate(st.session_state.generated_questions, start=1):
            st.markdown(render_question_card(idx, item), unsafe_allow_html=True)

        gen_df = pd.DataFrame(st.session_state.generated_questions)
        st.download_button(
            "Download as CSV",
            data=gen_df.to_csv(index=False).encode("utf-8"),
            file_name=f"predicted_{gen_subject.replace(' ', '_')}.csv",
            mime="text/csv",
        )

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Predicted Exam Questions - {gen_subject}", ln=True)
        for idx, item in enumerate(st.session_state.generated_questions, start=1):
            question_text = (
                f"{idx}. [{item.get('type', '')}] ({item.get('marks', 0)} marks) "
                f"{item.get('question', '')}"
            )
            safe_text = question_text.encode("latin-1", errors="replace").decode("latin-1")
            pdf.set_x(pdf.l_margin)
            usable_width = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.multi_cell(usable_width, 8, str(safe_text))
        st.download_button(
            "Download as PDF",
            data=bytes(pdf.output()),
            file_name=f"predicted_{gen_subject.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )


def _env_gemini_key() -> bool:
    """Backward-compatible check for the previous Gemini key name."""
    import os

    key = os.getenv("GEMINI_API_KEY", "")
    return bool(key and key not in {"your_gemini_key_here", "your_key_here"})


def _env_mistral_key() -> bool:
    """Check if a Mistral API key exists in environment."""
    import os

    key = os.getenv("MISTRAL_API_KEY", "")
    return bool(key and key not in {"your_mistral_key_here", "your_key_here"})


def page_similarity_search() -> None:
    """Render semantic similarity search page."""
    st.title("Similarity Search")
    if not run_full_analysis():
        return

    questions_df, _, embeddings = get_filtered_data()
    if questions_df.empty or embeddings is None:
        st.warning("No data available for the selected subject.")
        return

    tab_search, tab_similar = st.tabs(["Search", "Similar Questions Across Papers"])

    with tab_search:
        query = st.text_input("Enter a topic or question to search your uploaded exam papers")

        if query and st.button("Find Similar Questions"):
            agent_system = get_agent_system()
            materials_df = (
                st.session_state.subject_materials_df
                if st.session_state.subject_materials_df is not None
                else load_subject_materials()
            )
            subject_name = st.session_state.selected_subject_filter
            if subject_name == "All Subjects":
                subject_name = ""

            retrieval = agent_system.prediction_agent.retrieve_context(
                query=query,
                subject=subject_name,
                questions_df=questions_df,
                materials_df=materials_df,
                top_k=5,
            )

            st.subheader("Top 5 Most Similar Past Questions")
            if retrieval.matches.empty:
                st.info("No Pinecone matches returned yet. Upload more PDFs or check the index configuration.")
            else:
                for rank, (_, row) in enumerate(retrieval.matches.head(5).iterrows(), start=1):
                    st.markdown(
                        f"""
                        <div class="question-card">
                            <strong>#{rank} | Similarity: {row.get('rerank_score', row.get('score', 0.0)):.3f}</strong><br/>
                            <em>{row.get('subject', '')} — {row.get('topic_label', 'Unknown Topic')} ({row.get('metadata', {}).get('year', 'N/A')})</em><br/>
                            {row.get('text', '')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with tab_similar:
        st.caption(
            "Automatically finds questions that look alike across DIFFERENT uploaded PDFs — "
            "no search needed. Useful for spotting repeated or recycled questions between papers."
        )

        source_count = (
            questions_df["source_file"].nunique() if "source_file" in questions_df.columns else 0
        )
        if source_count < 2:
            st.info("Upload questions from at least 2 different PDFs to compare across papers.")
        else:
            threshold = st.slider(
                "Similarity threshold",
                min_value=0.70,
                max_value=0.99,
                value=0.85,
                step=0.01,
                help="Higher = only near-identical questions are shown. Lower = also catches "
                "loosely related questions.",
            )

            evaluator = ExamEvaluator()
            dup_df = evaluator.find_cross_paper_duplicates(
                questions_df, embeddings, threshold=threshold
            )

            if dup_df.empty:
                st.success(
                    f"No question pairs found above {threshold:.2f} similarity across "
                    "different PDFs."
                )
            else:
                pdfs_involved = pd.unique(dup_df[["source_a", "source_b"]].values.ravel())
                c1, c2 = st.columns(2)
                c1.metric("Similar question pairs found", len(dup_df))
                c2.metric("PDFs involved", len(pdfs_involved))

                for _, row in dup_df.iterrows():
                    st.markdown(
                        f"""
                        <div class="question-card">
                            <strong>Similarity: {row['similarity']:.3f}</strong><br/><br/>
                            <strong>📄 {row['source_a']}</strong>
                            <em>({row['subject_a']}, {row['year_a']})</em><br/>
                            {row['question_a']}<br/><br/>
                            <strong>📄 {row['source_b']}</strong>
                            <em>({row['subject_b']}, {row['year_b']})</em><br/>
                            {row['question_b']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.download_button(
                    "Download similar-question pairs (CSV)",
                    data=dup_df.to_csv(index=False).encode("utf-8"),
                    file_name="similar_questions_across_papers.csv",
                    mime="text/csv",
                )


def page_retrieval_evaluation() -> None:
    """Render retrieval quality metrics and ranking diagnostics."""
    st.title("Retrieval Evaluation")
    if not run_full_analysis():
        return

    questions_df, _, _ = get_filtered_data()
    materials_df = (
        st.session_state.subject_materials_df
        if st.session_state.subject_materials_df is not None
        else load_subject_materials()
    )

    if questions_df.empty:
        st.warning("No questions available to evaluate retrieval.")
        return

    subject_name = None if st.session_state.selected_subject_filter == "All Subjects" else st.session_state.selected_subject_filter
    agent_system = get_agent_system()
    summary, details = agent_system.evaluate_retrieval(
        questions_df,
        materials_df,
        subject=subject_name,
        sample_size=25,
        top_k=5,
    )

    st.session_state.retrieval_metrics = summary
    st.session_state.retrieval_details = details

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision@5", f"{summary.precision_at_k:.3f}")
    c2.metric("Recall@5", f"{summary.recall_at_k:.3f}")
    c3.metric("MRR", f"{summary.mean_reciprocal_rank:.3f}")
    c4.metric("nDCG@5", f"{summary.ndcg_at_k:.3f}")

    st.caption(f"Evaluated queries: {summary.evaluated_queries}")

    if not details.empty:
        st.subheader("Sample Retrieval Judgements")
        st.dataframe(
            details[["query_question", "retrieved_text", "rank_score", "topic_label", "query_topic", "is_relevant"]],
            use_container_width=True,
            hide_index=True,
        )

            if st.button("Find Repeated Questions Across Papers", type="primary"):
                evaluator = ExamEvaluator()
                with st.spinner("Computing pairwise similarity across all papers…"):
                    pairs_df = evaluator.find_cross_paper_duplicates(
                        questions_df, embeddings, threshold=threshold
                    )

                if pairs_df.empty:
                    st.info(
                        f"No question pairs found above **{threshold:.0%}** similarity. "
                        "Try lowering the threshold."
                    )
                else:
                    n_pairs = len(pairs_df)
                    pdf_pairs = (
                        pairs_df[["source_a", "source_b"]]
                        .drop_duplicates()
                        .shape[0]
                    )
                    exact_count = int((pairs_df["similarity"] >= 0.97).sum())
                    high_count = int(
                        ((pairs_df["similarity"] >= 0.90) & (pairs_df["similarity"] < 0.97)).sum()
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Similar Pairs Found", n_pairs)
                    m2.metric("PDF Pairs Compared", pdf_pairs)
                    m3.metric("Exact / Near-Identical", exact_count)
                    m4.metric("Very Similar", high_count)

                    # CSV export
                    export_cols = [
                        "similarity", "source_a", "year_a", "question_a",
                        "source_b", "year_b", "question_b",
                    ]
                    st.download_button(
                        "Download Results as CSV",
                        data=pairs_df[export_cols].to_csv(index=False).encode("utf-8"),
                        file_name="cross_paper_duplicates.csv",
                        mime="text/csv",
                    )

                    st.divider()

                    # Group by PDF pair and show expanders
                    grouped = pairs_df.groupby(["source_a", "source_b"], sort=False)
                    for (src_a, src_b), group in grouped:
                        n = len(group)
                        exact_in_group = int((group["similarity"] >= 0.97).sum())
                        label = (
                            f"📄 {src_a}  ↔  📄 {src_b} "
                            f"— {n} match{'es' if n != 1 else ''}"
                            + (f"  ({exact_in_group} exact)" if exact_in_group else "")
                        )
                        with st.expander(label, expanded=(n_pairs <= 20)):
                            for _, pair_row in group.iterrows():
                                _render_dup_card(pair_row)


def page_analytics_dashboard() -> None:
    """Render analytics overview dashboard."""
    st.title("Analytics Dashboard")
    if not run_full_analysis():
        return

    questions_df, topics_df, _ = get_filtered_data()
    if questions_df.empty:
        st.warning("No data for the selected subject filter.")
        return

    evaluator = ExamEvaluator()
    metrics = evaluator.get_overview_metrics(questions_df, topics_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exam PDFs", metrics["total_papers"])
    c2.metric("Questions Extracted", metrics["total_questions"])
    c3.metric("Topics Discovered", metrics["topics_discovered"])
    c4.metric("Subjects", len(get_subjects(questions_df)))

    year_df = evaluator.year_distribution(questions_df)
    if not year_df.empty:
        st.plotly_chart(
            px.bar(year_df, x="year", y="count", title="Year-wise Question Distribution"),
            use_container_width=True,
        )

    if "subject" in questions_df.columns:
        subj_df = questions_df.groupby("subject").size().reset_index(name="count")
        st.plotly_chart(
            px.bar(subj_df, x="subject", y="count", title="Questions by Subject"),
            use_container_width=True,
        )

    diff_df = evaluator.difficulty_distribution(questions_df)
    if not diff_df.empty:
        st.plotly_chart(
            evaluator.make_pie_chart(diff_df, "difficulty", "count", "Question Type Distribution"),
            use_container_width=True,
        )

    corr = evaluator.topic_correlation_matrix(questions_df)
    st.plotly_chart(
        evaluator.make_heatmap(corr, "Topic Correlation Heatmap"),
        use_container_width=True,
    )


def _silhouette_label(score: float) -> tuple[str, str]:
    """Return (emoji+text, colour) interpretation for a silhouette score."""
    if score >= 0.70:
        return "Strong clustering — topics are well separated", "normal"
    if score >= 0.50:
        return "Reasonable clustering — moderate topic overlap", "normal"
    if score >= 0.25:
        return "Weak clustering — topics overlap significantly", "off"
    return "Poor clustering — consider uploading more questions", "inverse"


def _compute_tsne(questions_df: pd.DataFrame, embeddings) -> pd.DataFrame:
    """Project embeddings to 2D via t-SNE and return a plot-ready DataFrame."""
    from sklearn.manifold import TSNE

    n = len(embeddings)
    perplexity = min(30, max(5, (n - 1) // 4))
    coords = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    ).fit_transform(embeddings)

    q = questions_df.reset_index(drop=True)
    return pd.DataFrame({
        "x": coords[:, 0].round(3),
        "y": coords[:, 1].round(3),
        "topic_label": q.get("topic_label", pd.Series(["Unknown"] * n)).fillna("Unknown").values,
        "year": q.get("year", pd.Series(["N/A"] * n)).astype(str).values,
        "subject": q.get("subject", pd.Series([""] * n)).astype(str).values,
        "question_preview": q["question_text"].astype(str).str[:100].values,
    })


def _compute_bleu_rouge(
    generated_questions: list,
    reference_questions: list[str],
) -> pd.DataFrame:
    """Return BLEU-1, BLEU-2, ROUGE-1, ROUGE-L per generated question."""
    if not generated_questions or not reference_questions:
        return pd.DataFrame()
    try:
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return pd.DataFrame()

    smoother = SmoothingFunction().method1
    ref_tokens = [word_tokenize(q.lower()) for q in reference_questions if q.strip()]

    rouge_scorer_obj = None
    try:
        from rouge_score import rouge_scorer as _rs
        rouge_scorer_obj = _rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    except ImportError:
        pass

    rows = []
    for item in generated_questions:
        gen_text = str(item.get("question", "")).strip()
        if not gen_text:
            continue
        hyp_tokens = word_tokenize(gen_text.lower())
        b1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother)
        b2 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
        row = {
            "Question": gen_text[:110] + ("…" if len(gen_text) > 110 else ""),
            "Type": item.get("type", ""),
            "Difficulty": item.get("difficulty", ""),
            "BLEU-1": round(b1, 4),
            "BLEU-2": round(b2, 4),
        }
        if rouge_scorer_obj is not None:
            best_r1 = best_rL = 0.0
            for ref in reference_questions[:30]:
                r = rouge_scorer_obj.score(ref, gen_text)
                if r["rouge1"].fmeasure > best_r1:
                    best_r1 = r["rouge1"].fmeasure
                    best_rL = r["rougeL"].fmeasure
            row["ROUGE-1"] = round(best_r1, 4)
            row["ROUGE-L"] = round(best_rL, 4)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _compute_tfidf_evaluation(questions_df: pd.DataFrame) -> dict:
    """Compute TF-IDF vectors, cluster them, and return evaluation data.

    Returns a dict with keys:
        silhouette_score, n_clusters, top_terms_per_topic, tfidf_matrix
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    texts = questions_df["question_text"].fillna("").tolist()
    n = len(texts)
    if n < 4:
        return {}

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Use same cluster-count formula as the main pipeline
    max_k = min(10, max(2, n // 4))
    min_k = min(3, max(2, n // 6))

    best_k, best_score, best_labels = min_k, -1.0, None
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(tfidf_matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(tfidf_matrix, labels, metric="cosine")
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    # Top TF-IDF terms per cluster
    top_terms: dict[int, list[str]] = {}
    if best_labels is not None:
        for cluster_id in sorted(set(best_labels)):
            mask = best_labels == cluster_id
            cluster_matrix = tfidf_matrix[mask]
            mean_tfidf = cluster_matrix.mean(axis=0).A1
            top_idx = mean_tfidf.argsort()[::-1][:5]
            top_terms[int(cluster_id)] = [feature_names[i] for i in top_idx]

    # Cosine similarity distribution (sample up to 200 pairs)
    dense = tfidf_matrix.toarray()
    sim_matrix = cosine_similarity(dense)
    upper = sim_matrix[np.triu_indices(n, k=1)]

    return {
        "silhouette_score": round(float(best_score), 4),
        "n_clusters": best_k,
        "top_terms_per_topic": top_terms,
        "similarity_distribution": upper.tolist(),
        "labels": best_labels,
    }


def _get_tsne_cached(questions_df: pd.DataFrame, embeddings) -> pd.DataFrame:
    """Return t-SNE DataFrame, cached in session_state to avoid recomputing."""
    cache_key = f"_tsne_{len(questions_df)}_{embeddings.shape[0]}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = _compute_tsne(questions_df, embeddings)
    return st.session_state[cache_key]


def page_evaluation_metrics() -> None:
    """Render evaluation and performance metrics page."""
    st.title("Evaluation Metrics")
    st.caption(
        "Quantitative assessment of clustering quality, embedding structure, "
        "and generated question similarity."
    )

    if not run_full_analysis():
        return

    questions_df, topics_df, embeddings = get_filtered_data()
    if questions_df.empty or embeddings is None:
        st.warning("No data available for the selected subject filter.")
        return

    tab_cluster, tab_tsne, tab_bleu, tab_embed = st.tabs([
        "Clustering Quality",
        "Embedding Visualisation (t-SNE)",
        "Generation Quality (BLEU / ROUGE)",
        "Embedding Comparison (TF-IDF vs Sentence)",
    ])

    # ── Tab 1: Clustering Quality ────────────────────────────────────────────
    with tab_cluster:
        st.subheader("KMeans Clustering Quality")

        sil_score = float(st.session_state.get("silhouette_score", 0.0))

        # Recompute if filtered view differs from full dataset
        if "topic_id" in questions_df.columns and len(set(questions_df["topic_id"])) >= 2:
            from sklearn.metrics import silhouette_score as _sil
            try:
                sil_score = float(_sil(embeddings, questions_df["topic_id"].values))
            except Exception:
                pass

        interp, delta_color = _silhouette_label(sil_score)
        n_topics = int(topics_df["topic_label"].nunique()) if not topics_df.empty else 0
        avg_q = int(len(questions_df) / n_topics) if n_topics else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette Score", f"{sil_score:.4f}", delta=interp, delta_color=delta_color)
        c2.metric("Topics (Clusters)", n_topics)
        c3.metric("Avg Questions / Topic", avg_q)

        st.info(
            "**Silhouette Score** ranges from -1 to 1.  "
            "Values above **0.50** indicate well-separated topic clusters.  "
            "This score was maximised automatically over 3–10 clusters using the "
            "silhouette criterion during the KMeans fitting step."
        )

        st.divider()
        st.subheader("Topic-level Breakdown")
        if not topics_df.empty:
            display = topics_df[["topic_label", "question_count", "trend"]].copy()
            display["% of questions"] = (
                display["question_count"] / display["question_count"].sum() * 100
            ).round(1).astype(str) + "%"
            st.dataframe(display, use_container_width=True, hide_index=True)

            # Bar chart: topic sizes
            fig = px.bar(
                display.sort_values("question_count", ascending=True),
                x="question_count",
                y="topic_label",
                orientation="h",
                title="Questions per Topic",
                labels={"question_count": "Questions", "topic_label": "Topic"},
                template="plotly_dark",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: t-SNE Embedding Visualisation ────────────────────────────────
    with tab_tsne:
        st.subheader("Question Embedding Space — t-SNE Projection")
        st.write(
            "Each dot is one exam question projected from a 384-dimensional semantic "
            "embedding to 2D.  Questions that cluster together share similar meaning.  "
            "Colours represent the automatically discovered topic groups."
        )
        st.caption(
            "Note: t-SNE preserves local neighbourhood structure, not global distances.  "
            "Cluster sizes and inter-cluster distances are not directly comparable."
        )

        n = len(embeddings)
        if n < 6:
            st.warning("At least 6 questions are needed for a meaningful t-SNE plot.")
        else:
            if st.button("Generate t-SNE Plot", type="primary"):
                with st.spinner(f"Projecting {n} questions to 2D — this may take ~10s…"):
                    tsne_df = _get_tsne_cached(questions_df, embeddings)

                fig = px.scatter(
                    tsne_df,
                    x="x",
                    y="y",
                    color="topic_label",
                    hover_data={"x": False, "y": False,
                                "question_preview": True, "year": True, "subject": True},
                    title="Question Embeddings — t-SNE 2D",
                    template="plotly_dark",
                    labels={"topic_label": "Topic", "question_preview": "Question"},
                )
                fig.update_traces(marker=dict(size=7, opacity=0.8))
                fig.update_layout(
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="v", x=1.01, y=0.5),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download t-SNE data
                st.download_button(
                    "Download t-SNE Coordinates (CSV)",
                    data=tsne_df.to_csv(index=False).encode("utf-8"),
                    file_name="tsne_embeddings.csv",
                    mime="text/csv",
                )
            else:
                st.info("Click **Generate t-SNE Plot** above to visualise the embedding space.")

    # ── Tab 3: BLEU / ROUGE ─────────────────────────────────────────────────
    with tab_bleu:
        st.subheader("Generated Question Quality — BLEU & ROUGE")
        st.write(
            "Measures how closely the **Gemini-generated** questions resemble real past "
            "exam questions in vocabulary and phrasing.  "
            "Go to **Question Predictions**, generate questions, then return here."
        )
        st.info(
            "**Interpreting scores:** BLEU and ROUGE measure n-gram overlap with "
            "reference questions.  For question *generation*, moderate scores "
            "**(0.10–0.40)** are ideal — they show the questions follow exam style "
            "without being verbatim copies.  Very high scores (>0.60) would indicate "
            "repetition; very low scores (<0.05) may mean off-topic output."
        )

        generated = st.session_state.get("generated_questions", [])
        gen_topic = st.session_state.get("gen_topic_for_bleu", "")
        gen_subject = st.session_state.get("gen_subject_for_bleu", "")

        if not generated:
            st.warning("No generated questions yet. Go to **Question Predictions** and generate some first.")
        else:
            # Build reference corpus from same topic/subject
            ref_df = questions_df.copy()
            all_refs = ref_df["question_text"].dropna().tolist()
            if gen_topic:
                topic_refs = ref_df[ref_df["topic_label"] == gen_topic]["question_text"].tolist()
                # Need ≥10 topic questions for BLEU to be meaningful; fall back to full corpus.
                reference_questions = topic_refs if len(topic_refs) >= 10 else all_refs
            else:
                reference_questions = all_refs

            st.caption(
                f"Comparing **{len(generated)} generated question(s)** against "
                f"**{len(reference_questions)} reference question(s)** "
                f"from topic: *{gen_topic or 'all'}* / subject: *{gen_subject or 'all'}*."
            )

            with st.spinner("Computing BLEU & ROUGE scores…"):
                scores_df = _compute_bleu_rouge(generated, reference_questions)

            if scores_df.empty:
                st.error("Could not compute scores. Ensure NLTK data is downloaded.")
            else:
                score_cols = [c for c in ["BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-L"] if c in scores_df.columns]

                # Summary metrics row
                avg = scores_df[score_cols].mean()
                cols = st.columns(len(score_cols))
                for col, metric in zip(cols, score_cols):
                    col.metric(f"Avg {metric}", f"{avg[metric]:.4f}")

                st.divider()
                st.dataframe(
                    scores_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Question": st.column_config.TextColumn(width="large"),
                        "BLEU-1": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                        "BLEU-2": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                        "ROUGE-1": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                        "ROUGE-L": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                    },
                )

                st.download_button(
                    "Download Scores as CSV",
                    data=scores_df.to_csv(index=False).encode("utf-8"),
                    file_name="bleu_rouge_scores.csv",
                    mime="text/csv",
                )

    # ── Tab 4: Embedding Comparison ──────────────────────────────────────────
    with tab_embed:
        st.subheader("Embedding Method Comparison: TF-IDF vs Sentence Transformers")
        st.write(
            "Compares two fundamentally different ways of turning exam questions into "
            "vectors.  **TF-IDF** uses word frequency statistics (fast, interpretable).  "
            "**Sentence Transformers** (`all-MiniLM-L6-v2`) use deep learning to capture "
            "semantic meaning even when different words express the same idea."
        )

        if st.button("Run Embedding Comparison", type="primary"):
            with st.spinner("Computing TF-IDF vectors and clustering…"):
                tfidf_result = _compute_tfidf_evaluation(questions_df)

            if not tfidf_result:
                st.warning("Need at least 4 questions to compare embedding methods.")
            else:
                # ── Silhouette comparison ────────────────────────────────────
                st.subheader("Clustering Quality — Silhouette Score")

                sil_sent = 0.0
                if "topic_id" in questions_df.columns and len(set(questions_df["topic_id"])) >= 2:
                    from sklearn.metrics import silhouette_score as _sil
                    try:
                        sil_sent = float(_sil(embeddings, questions_df["topic_id"].values))
                    except Exception:
                        pass

                sil_tfidf = tfidf_result["silhouette_score"]
                n_clust_tfidf = tfidf_result["n_clusters"]
                n_clust_sent = int(questions_df["topic_id"].nunique()) if "topic_id" in questions_df.columns else 0

                c1, c2 = st.columns(2)
                with c1:
                    st.metric(
                        "Sentence Transformer Silhouette",
                        f"{sil_sent:.4f}",
                        delta=f"{n_clust_sent} clusters",
                        delta_color="off",
                    )
                with c2:
                    st.metric(
                        "TF-IDF Silhouette",
                        f"{sil_tfidf:.4f}",
                        delta=f"{n_clust_tfidf} clusters",
                        delta_color="off",
                    )

                # Bar chart comparison
                cmp_df = pd.DataFrame({
                    "Method": ["TF-IDF", "Sentence Transformer"],
                    "Silhouette Score": [sil_tfidf, sil_sent],
                    "Clusters": [n_clust_tfidf, n_clust_sent],
                })
                fig_cmp = px.bar(
                    cmp_df,
                    x="Method",
                    y="Silhouette Score",
                    color="Method",
                    text="Silhouette Score",
                    title="Silhouette Score: TF-IDF vs Sentence Transformer",
                    template="plotly_dark",
                    color_discrete_map={
                        "TF-IDF": "#f4a261",
                        "Sentence Transformer": "#4cc9f0",
                    },
                )
                fig_cmp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig_cmp.update_layout(
                    yaxis=dict(range=[0, 1]),
                    showlegend=False,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                # Interpretation
                winner = "Sentence Transformer" if sil_sent >= sil_tfidf else "TF-IDF"
                diff = abs(sil_sent - sil_tfidf)
                if diff < 0.02:
                    interp = "Both methods produce **similar clustering quality** on your dataset."
                elif winner == "Sentence Transformer":
                    interp = (
                        f"**Sentence Transformer wins** by {diff:.4f} — it captures semantic "
                        "similarity better (e.g. 'compute the hash' and 'find the digest' cluster "
                        "together even though the words differ)."
                    )
                else:
                    interp = (
                        f"**TF-IDF wins** by {diff:.4f} — your questions use very consistent "
                        "keyword patterns so word-frequency statistics are sufficient."
                    )
                st.info(interp)

                st.divider()

                # ── Top TF-IDF terms per topic ───────────────────────────────
                st.subheader("Top TF-IDF Keywords per Topic")
                st.caption(
                    "These are the highest-weighted terms for each topic cluster "
                    "discovered by TF-IDF vectorization — useful for interpreting "
                    "what each cluster is actually about."
                )
                top_terms = tfidf_result.get("top_terms_per_topic", {})
                if top_terms:
                    term_rows = [
                        {"Topic": f"Topic {tid + 1}", "Top Keywords": " · ".join(terms)}
                        for tid, terms in sorted(top_terms.items())
                    ]
                    st.dataframe(
                        pd.DataFrame(term_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.divider()

                # ── Cosine similarity distribution ───────────────────────────
                st.subheader("Pairwise Cosine Similarity Distribution")
                st.caption(
                    "Shows how similar questions are to each other under each method.  "
                    "A tighter distribution near 0 means questions are well-separated; "
                    "a peak near 1 means many questions are very similar."
                )
                sim_vals = tfidf_result.get("similarity_distribution", [])
                if sim_vals:
                    import numpy as np
                    sim_df = pd.DataFrame({
                        "Cosine Similarity": sim_vals,
                        "Method": ["TF-IDF"] * len(sim_vals),
                    })
                    fig_hist = px.histogram(
                        sim_df,
                        x="Cosine Similarity",
                        nbins=40,
                        title="TF-IDF Pairwise Similarity Distribution",
                        template="plotly_dark",
                        color_discrete_sequence=["#f4a261"],
                    )
                    fig_hist.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.caption(
                        f"Median similarity: **{float(np.median(sim_vals)):.4f}** | "
                        f"Max: **{float(np.max(sim_vals)):.4f}** | "
                        f"Pairs above 0.85: **{sum(1 for s in sim_vals if s >= 0.85)}**"
                    )

                # Download comparison summary
                st.download_button(
                    "Download Comparison Summary (CSV)",
                    data=cmp_df.to_csv(index=False).encode("utf-8"),
                    file_name="embedding_comparison.csv",
                    mime="text/csv",
                )
        else:
            st.info(
                "Click **Run Embedding Comparison** to vectorise your questions with TF-IDF "
                "and compare against the Sentence Transformer embeddings."
            )


def main() -> None:
    """Main Streamlit application entry point."""
    init_session_state()
    render_sidebar()

    page = st.session_state.get("current_page", "Upload & Process")
    pages = {
        "Upload & Process": page_upload_process,
        "Topic Analysis": page_topic_analysis,
        "Question Predictions": page_question_predictions,
        "Similarity Search": page_similarity_search,
        "Retrieval Evaluation": page_retrieval_evaluation,
        "Analytics Dashboard": page_analytics_dashboard,
        "Evaluation Metrics": page_evaluation_metrics,
    }
    pages[page]()


if __name__ == "__main__":
    main()