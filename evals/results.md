# Retrieval eval

866 real job listings (frozen snapshot), 20 cases. Labels are rule-based (title regex), so compare the two columns rather than reading them as absolute quality.

| Metric (mean over cases) | Semantic (MiniLM) | Keyword baseline |
|---|---|---|
| Precision@5 | 0.54 | 0.17 |
| Recall@10 | 0.45 | 0.12 |
| MRR | 0.72 | 0.29 |

| Case | Relevant jobs | P@5 semantic | P@5 keyword |
|---|---|---|---|
| backend developer | 60 | 0.2 | 0.2 |
| frontend developer | 23 | 0.8 | 0.4 |
| machine learning | 73 | 1.0 | 0.0 |
| data analyst | 39 | 0.8 | 0.2 |
| devops / cloud | 31 | 1.0 | 0.8 |
| security | 18 | 1.0 | 0.8 |
| sales | 106 | 0.4 | 0.2 |
| marketing | 47 | 0.8 | 0.0 |
| customer support | 30 | 0.6 | 0.2 |
| product manager | 40 | 0.6 | 0.0 |
| designer | 19 | 0.2 | 0.0 |
| finance / accounting | 23 | 1.0 | 0.2 |
| recruiter | 15 | 0.2 | 0.0 |
| project manager | 17 | 0.2 | 0.0 |
| QA / testing | 16 | 0.4 | 0.0 |
| legal | 13 | 0.4 | 0.2 |
| writer / editor | 14 | 0.4 | 0.0 |
| healthcare | 11 | 0.2 | 0.2 |
| mobile developer | 7 | 0.4 | 0.0 |
| data engineer | 14 | 0.2 | 0.0 |
