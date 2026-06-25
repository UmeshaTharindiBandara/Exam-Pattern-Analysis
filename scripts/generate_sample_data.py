"""Generate optional demo exam question data (development utility only).

The main application uses uploaded PDFs only. This script is not used by the Streamlit app.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TOPICS = [
    "Machine Learning",
    "Databases",
    "Algorithms",
    "Networks",
    "Software Engineering",
]

QUESTION_BANK: dict[str, list[str]] = {
    "Machine Learning": [
        "Explain the bias-variance tradeoff in supervised learning models.",
        "Compare gradient descent and stochastic gradient descent for neural network training.",
        "Define overfitting and describe three techniques to reduce it.",
        "Which evaluation metric is most appropriate for imbalanced classification?",
        "Discuss the role of regularization in linear regression.",
        "Explain how convolutional neural networks extract spatial features.",
        "Describe the working of the k-means clustering algorithm.",
        "What is transfer learning and when should it be used?",
        "Explain precision, recall, and F1-score with examples.",
        "Discuss ethical concerns in deploying machine learning systems.",
    ],
    "Databases": [
        "Explain ACID properties in relational database systems.",
        "Compare SQL and NoSQL databases with suitable use cases.",
        "Define normalization and explain up to Third Normal Form (3NF).",
        "Describe indexing strategies and their impact on query performance.",
        "What is a transaction isolation level? Explain serializable isolation.",
        "Explain entity-relationship modeling with a practical example.",
        "Discuss CAP theorem in distributed database design.",
        "Describe query optimization techniques used by modern DBMS engines.",
        "Explain the difference between clustered and non-clustered indexes.",
        "What are database triggers and stored procedures?",
    ],
    "Algorithms": [
        "Analyze the time complexity of merge sort and quicksort.",
        "Explain Dijkstra's algorithm with a step-by-step example.",
        "Compare dynamic programming and greedy algorithm design paradigms.",
        "Describe how hash tables handle collisions.",
        "Prove that binary search runs in O(log n) time.",
        "Explain the difference between BFS and DFS graph traversals.",
        "Design an algorithm to detect cycles in a directed graph.",
        "Discuss amortized analysis with an example.",
        "Explain NP-completeness and provide one NP-complete problem.",
        "Describe divide-and-conquer strategy using merge sort.",
    ],
    "Networks": [
        "Explain the TCP/IP protocol stack and its layers.",
        "Compare TCP and UDP transport protocols.",
        "Describe how DNS resolution works from client to authoritative server.",
        "What is the purpose of ARP in local area networks?",
        "Explain routing algorithms used in modern IP networks.",
        "Discuss security threats in wireless networks and mitigation methods.",
        "Describe HTTP/2 improvements over HTTP/1.1.",
        "Explain NAT and its role in IPv4 address conservation.",
        "What is a subnet mask and how is CIDR notation used?",
        "Discuss quality of service (QoS) mechanisms in network traffic management.",
    ],
    "Software Engineering": [
        "Explain the Agile Scrum development process.",
        "Compare waterfall and iterative software development models.",
        "Describe SOLID principles with examples.",
        "What is continuous integration and why is it important?",
        "Discuss test-driven development and its benefits.",
        "Explain microservices architecture and associated trade-offs.",
        "Describe UML class diagrams for object-oriented design.",
        "What are design patterns? Explain Singleton and Observer patterns.",
        "Discuss code refactoring techniques and when to apply them.",
        "Explain software requirements specification and traceability.",
    ],
}

YEARS = list(range(2019, 2025))
SUBJECT = "Computer Science"


def generate_sample_questions(total: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate fake exam questions across predefined topics and years.

    Args:
        total: Number of questions to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with question records.
    """
    random.seed(seed)
    records: list[dict[str, str | int]] = []
    question_id = 1

    per_topic = total // len(TOPICS)
    remainder = total % len(TOPICS)

    for topic_idx, topic in enumerate(TOPICS):
        count = per_topic + (1 if topic_idx < remainder else 0)
        pool = QUESTION_BANK[topic]
        selected = random.sample(pool, k=min(count, len(pool)))

        for text in selected:
            year = random.choice(YEARS)
            records.append(
                {
                    "question_id": f"Q{question_id:03d}",
                    "question_text": text,
                    "year": year,
                    "subject": SUBJECT,
                    "marks": random.choice([2, 5, 8, 10, 12, 15]),
                    "source_file": f"sample_{topic.lower().replace(' ', '_')}_{year}.pdf",
                    "topic_label": topic,
                }
            )
            question_id += 1

    while len(records) < total:
        topic = random.choice(TOPICS)
        text = random.choice(QUESTION_BANK[topic])
        records.append(
            {
                "question_id": f"Q{question_id:03d}",
                "question_text": text,
                "year": random.choice(YEARS),
                "subject": SUBJECT,
                "marks": random.choice([2, 5, 8, 10, 12, 15]),
                "source_file": f"sample_{topic.lower().replace(' ', '_')}.pdf",
                "topic_label": topic,
            }
        )
        question_id += 1

    return pd.DataFrame(records)


def main() -> None:
    """CLI entry point for sample data generation."""
    parser = argparse.ArgumentParser(description="Generate sample exam question dataset.")
    parser.add_argument("--total", type=int, default=50, help="Number of questions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DIR / "sample_questions.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_sample_questions(total=args.total, seed=args.seed)
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} sample questions -> {output_path}")


if __name__ == "__main__":
    main()
