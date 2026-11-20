# Retrieval experiments

866 listings, 20 cases, title-regex labels (see evals/run.py). One relevant job more or less in a case's top 5 moves mean precision@5 by 0.01, so differences of a few hundredths are noise.

| Method | Precision@5 | Recall@10 | MRR |
|---|---|---|---|
| all-MiniLM-L6-v2 (what the app uses) | 0.54 | 0.45 | 0.72 |
| bge-small-en-v1.5 | 0.48 | 0.42 | 0.62 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.56 | 0.42 | 0.79 |
| BM25 keyword search | 0.39 | 0.29 | 0.64 |
| Hybrid: MiniLM + BM25, RRF, equal weight | 0.50 | 0.43 | 0.77 |
| Hybrid: MiniLM + BM25, RRF, vector counted twice | 0.54 | 0.46 | 0.81 |
