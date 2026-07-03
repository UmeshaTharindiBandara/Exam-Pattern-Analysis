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
    }
    pages[page]()


if __name__ == "__main__":
    main()