# Rerank eval

866 real job listings (frozen snapshot), 20 cases scored, model `openai/gpt-oss-120b`. Vector search retrieves 12 jobs per case; the LLM reorders them. Labels are rule-based (title regex), so small differences are noise.

| Metric (mean over cases) | Vector order | After LLM rerank | Ceiling for this pool |
|---|---|---|---|
| Precision@5 | 0.54 | 0.73 | 0.79 |
| MRR | 0.72 | 0.91 | - |

Precision@5 per case: rerank better in 11, worse in 1, the same in 8.

| Case | Relevant in pool | P@5 vector | P@5 LLM | Ceiling | Jobs the LLM ranked |
|---|---|---|---|---|---|
| backend developer | 4 | 0.2 | 0.4 | 0.8 | 11/12 |
| frontend developer | 5 | 0.8 | 1.0 | 1.0 | 12/12 |
| machine learning | 8 | 1.0 | 0.6 | 1.0 | 12/12 |
| data analyst | 5 | 0.8 | 0.8 | 1.0 | 12/12 |
| devops / cloud | 10 | 1.0 | 1.0 | 1.0 | 12/12 |
| security | 8 | 1.0 | 1.0 | 1.0 | 12/12 |
| sales | 6 | 0.4 | 1.0 | 1.0 | 12/12 |
| marketing | 9 | 0.8 | 1.0 | 1.0 | 12/12 |
| customer support | 4 | 0.6 | 0.6 | 0.8 | 12/12 |
| product manager | 8 | 0.6 | 1.0 | 1.0 | 12/12 |
| designer | 2 | 0.2 | 0.4 | 0.4 | 12/12 |
| finance / accounting | 6 | 1.0 | 1.0 | 1.0 | 12/12 |
| recruiter | 1 | 0.2 | 0.2 | 0.2 | 12/12 |
| project manager | 1 | 0.2 | 0.2 | 0.2 | 12/12 |
| QA / testing | 6 | 0.4 | 1.0 | 1.0 | 12/12 |
| legal | 4 | 0.4 | 0.8 | 0.8 | 12/12 |
| writer / editor | 4 | 0.4 | 0.8 | 0.8 | 12/12 |
| healthcare | 1 | 0.2 | 0.2 | 0.2 | 12/12 |
| mobile developer | 5 | 0.4 | 1.0 | 1.0 | 12/12 |
| data engineer | 3 | 0.2 | 0.6 | 0.6 | 12/12 |
