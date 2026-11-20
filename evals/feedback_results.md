# Feedback eval

866 listings, 20 cases, simulated feedback on the first 5 results (save up to 2 relevant, hide up to 2 irrelevant), then the same query again.

| Mean over cases | Search again without feedback | Search shifted by feedback |
|---|---|---|
| Precision@5 on the next page | 0.41 | 0.52 |

Per case: better in 9, worse in 4, the same in 7.

| Case | Saved | Hidden | P@5 without | P@5 with |
|---|---|---|---|---|
| backend developer | 1 | 2 | 0.4 | 0.6 |
| frontend developer | 2 | 1 | 0.4 | 0.4 |
| machine learning | 2 | 0 | 0.6 | 0.8 |
| data analyst | 2 | 1 | 0.6 | 0.4 |
| devops / cloud | 2 | 0 | 0.8 | 0.8 |
| security | 2 | 0 | 1.0 | 0.8 |
| sales | 2 | 2 | 0.4 | 0.8 |
| marketing | 2 | 1 | 1.0 | 0.8 |
| customer support | 2 | 2 | 0.4 | 0.2 |
| product manager | 2 | 2 | 0.6 | 1.0 |
| designer | 1 | 2 | 0.0 | 0.4 |
| finance / accounting | 2 | 0 | 0.8 | 0.8 |
| recruiter | 1 | 2 | 0.0 | 0.2 |
| project manager | 1 | 2 | 0.0 | 0.4 |
| QA / testing | 2 | 2 | 0.6 | 1.0 |
| legal | 2 | 2 | 0.2 | 0.2 |
| writer / editor | 2 | 2 | 0.2 | 0.2 |
| healthcare | 1 | 2 | 0.0 | 0.0 |
| mobile developer | 2 | 2 | 0.2 | 0.2 |
| data engineer | 1 | 2 | 0.0 | 0.4 |
