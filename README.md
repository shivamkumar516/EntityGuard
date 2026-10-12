# EntityGuard

Scalable Entity Resolution and Duplicate Detection

\# EntityGuard: Scalable Entity Resolution Across Bibliographic Databases



\## 1. Problem Overview

Cross-matching DBLP and Google Scholar benchmark datasets (2,616 DBLP × 64,263 Scholar records = 168.1M potential pairs) using multi-pass blocking and string similarity algorithms.



\## 2. Team Division of Responsibilities

\- \*\*Student 1 (String \& Matching Algorithms):\*\* Text normalization, DP Levenshtein distance, token Jaccard, character n-grams.

\- \*\*Student 2 (Database \& Blocking):\*\* Relational schema setup (SQLite), indexing, multi-pass candidate blocking, search-space reduction.

\- \*\*Student 3 (Evaluation \& ML):\*\* Baseline exact matcher, precision/recall/F1 evaluation vs. gold mapping, human-in-the-loop review UI.



\## 3. Four-Week Milestones

\- \*\*Week 1:\*\* Data ingestion, text normalization, SQLite storage, and exact-match baseline.

\- \*\*Week 2:\*\* Levenshtein DP, Jaccard, and multi-attribute weighted matching.

\- \*\*Week 3:\*\* Year and title-token blocking strategies (targeting >99% pair reduction).

\- \*\*Week 4:\*\* Evaluation benchmarking, error analysis, and review dashboard.

