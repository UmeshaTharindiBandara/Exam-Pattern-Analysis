"""Analytics and evaluation helpers for exam pattern analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ExamEvaluator:
    """Compute dashboard metrics and visualization-ready datasets."""

    @staticmethod
    def get_overview_metrics(
        questions_df: pd.DataFrame,
        topics_df: pd.DataFrame | None = None,
    ) -> dict[str, int]:
        """Compute high-level analytics metrics.

        Args:
            questions_df: Annotated questions dataframe.
            topics_df: Topic summary dataframe.

        Returns:
            Dictionary of metric values.
        """
        papers = 0
        if "source_file" in questions_df.columns:
            papers = int(questions_df["source_file"].nunique())
        elif "subject" in questions_df.columns and "year" in questions_df.columns:
            papers = int(questions_df[["subject", "year"]].drop_duplicates().shape[0])

        topics_count = 0
        if topics_df is not None and not topics_df.empty:
            topics_count = int(len(topics_df))
        elif "topic_label" in questions_df.columns:
            topics_count = int(questions_df["topic_label"].nunique())

        return {
            "total_papers": papers,
            "total_questions": int(len(questions_df)),
            "topics_discovered": topics_count,
        }

    @staticmethod
    def year_distribution(questions_df: pd.DataFrame) -> pd.DataFrame:
        """Build year-wise question count dataset.

        Args:
            questions_df: Questions dataframe.

        Returns:
            Aggregated year counts.
        """
        if "year" not in questions_df.columns:
            return pd.DataFrame(columns=["year", "count"])
        return (
            questions_df.groupby("year")
            .size()
            .reset_index(name="count")
            .sort_values("year")
        )

    @staticmethod
    def difficulty_distribution(questions_df: pd.DataFrame) -> pd.DataFrame:
        """Build difficulty distribution based on question type proxy.

        Args:
            questions_df: Questions dataframe with question_type column.

        Returns:
            Distribution dataframe.
        """
        if "question_type" not in questions_df.columns:
            return pd.DataFrame({"difficulty": ["Unknown"], "count": [len(questions_df)]})

        mapping = {
            "MCQ": "Easy",
            "short_answer": "Medium",
            "calculation": "Hard",
            "essay": "Hard",
            "unknown": "Medium",
        }
        temp = questions_df.copy()
        temp["difficulty"] = temp["question_type"].map(mapping).fillna("Medium")
        return temp.groupby("difficulty").size().reset_index(name="count")

    @staticmethod
    def topic_correlation_matrix(questions_df: pd.DataFrame) -> pd.DataFrame:
        """Build topic co-occurrence matrix by year.

        Args:
            questions_df: Annotated questions dataframe.

        Returns:
            Correlation matrix dataframe.
        """
        if questions_df.empty or "topic_label" not in questions_df.columns:
            return pd.DataFrame()

        pivot = pd.crosstab(questions_df["year"], questions_df["topic_label"])
        if pivot.shape[1] < 2:
            return pd.DataFrame()
        return pivot.corr(numeric_only=True)

    @staticmethod
    def top_topics_bar_data(topics_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Prepare top topic frequency data for bar charts.

        Args:
            topics_df: Topic summary dataframe.
            top_n: Number of topics to include.

        Returns:
            Filtered topic dataframe.
        """
        if topics_df.empty:
            return topics_df
        return topics_df.nlargest(top_n, "question_count")

    @staticmethod
    def topic_trend_line_data(topics_df: pd.DataFrame) -> pd.DataFrame:
        """Expand topic yearly frequencies into long-format chart data.

        Args:
            topics_df: Topic summary dataframe.

        Returns:
            Long-format dataframe with year and count columns.
        """
        records: list[dict[str, Any]] = []
        for _, row in topics_df.iterrows():
            freq = row.get("frequency_by_year", {})
            if isinstance(freq, str):
                continue
            for year, count in freq.items():
                records.append(
                    {
                        "topic_label": row["topic_label"],
                        "year": int(year),
                        "count": int(count),
                    }
                )
        return pd.DataFrame(records)

    @staticmethod
    def make_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        """Create a Plotly bar chart."""
        fig = px.bar(df, x=x, y=y, title=title, template="plotly_dark")
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def make_line_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> go.Figure:
        """Create a Plotly line chart."""
        fig = px.line(df, x=x, y=y, color=color, title=title, template="plotly_dark")
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def make_pie_chart(df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
        """Create a Plotly pie chart."""
        fig = px.pie(df, names=names, values=values, title=title, template="plotly_dark")
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def make_heatmap(corr_df: pd.DataFrame, title: str) -> go.Figure:
        """Create a Plotly heatmap from a correlation matrix."""
        if corr_df.empty:
            fig = go.Figure()
            fig.update_layout(title=title, template="plotly_dark")
            return fig

        fig = px.imshow(
            corr_df,
            text_auto=".2f",
            title=title,
            template="plotly_dark",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig
