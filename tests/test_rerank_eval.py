from app.llm import LLMError
from evals.rerank import compare, precision_at_k, reciprocal_rank, report

from .conftest import FakeLLM

CASE = {"name": "backend", "query": "python fastapi postgresql rest apis"}


def test_metrics():
    ranked = ["a", "b", "c", "d"]
    assert precision_at_k(ranked, {"b", "d"}, 2) == 0.5
    assert reciprocal_rank(ranked, {"c"}) == 1 / 3
    assert reciprocal_rank(ranked, {"zzz"}) == 0.0


def test_compare_scores_vector_order_and_llm_order_separately(index):
    # Find what vector search returns, then script the model to move the last job to the top.
    vector_order = [job.id for job, _ in index.search(CASE["query"], limit=3)]
    last = vector_order[-1]
    llm = FakeLLM(json_replies=[{"ranking": [{"id": "J3", "score": 90}, {"id": "J1", "score": 50}, {"id": "J2", "score": 10}]}])

    rows, errors = compare(index, [CASE], {"backend": {last}}, llm, pool=3, k=1)

    assert errors == {}
    row = rows["backend"]
    assert row["relevant_in_pool"] == 1
    assert (row["p_vector"], row["p_llm"], row["p_ceiling"]) == (0.0, 1.0, 1.0)
    assert (row["mrr_vector"], row["mrr_llm"]) == (1 / 3, 1.0)
    assert row["ranked_by_llm"] == 3


def test_compare_leaves_out_cases_where_the_llm_fails(index):
    llm = FakeLLM(json_replies=[LLMError("rate limited")])
    rows, errors = compare(index, [CASE], {"backend": {"t:py"}}, llm, pool=3, k=1)
    assert rows == {}
    assert "rate limited" in errors["backend"]


def test_report_counts_better_and_worse_cases():
    row = {"relevant_in_pool": 2, "p_ceiling": 0.4, "mrr_vector": 0.5, "mrr_llm": 1.0, "ranked_by_llm": 12}
    rows = {"a": {**row, "p_vector": 0.2, "p_llm": 0.4}, "b": {**row, "p_vector": 0.4, "p_llm": 0.2}}
    text = report(rows, {"c": "boom"}, model="m", jobs=10, pool=12, k=5)
    assert "| Precision@5 | 0.30 | 0.30 | 0.40 |" in text
    assert "rerank better in 1, worse in 1, the same in 0" in text
    assert "Left out because the LLM call failed (1): c." in text
