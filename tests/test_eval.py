from unittest.mock import MagicMock, patch

from rag.eval import compute_expected_match, evaluate


def test_compute_expected_match_all_match():
    retrieved = ["docs/install.md", "docs/config.md"]
    expected = ["install.md", "config.md"]
    result = compute_expected_match(retrieved, expected)
    assert result.startswith("2/2 matched")


def test_compute_expected_match_partial():
    retrieved = ["docs/install.md", "docs/other.md"]
    expected = ["install.md"]
    result = compute_expected_match(retrieved, expected)
    assert result.startswith("1/2 matched")


def test_compute_expected_match_none():
    retrieved = ["docs/other.md"]
    expected = ["install.md"]
    result = compute_expected_match(retrieved, expected)
    assert result.startswith("0/1 matched")


def test_compute_expected_match_empty_expected():
    retrieved = ["docs/install.md"]
    expected = []
    result = compute_expected_match(retrieved, expected)
    assert result.startswith("0/1 matched")


def test_evaluate_without_key_facts():
    obj = MagicMock()
    obj.properties = {"source_path": "docs/install.md"}
    result = evaluate("query", "response", [obj], ["install.md"], [])
    assert "expected_match" in result
    assert "recall_score" not in result


@patch("rag.eval.score_recall", return_value=("3/5", [True, True, True, False, False], "missed Docker and K8s details"))
@patch("rag.eval.check_retrieval", return_value=[True, True, True, True, False])
def test_evaluate_with_key_facts(mock_retrieval, mock_recall):
    obj = MagicMock()
    obj.properties = {"source_path": "docs/install.md", "text": "some chunk text"}
    key_facts = ["fact1", "fact2", "fact3", "fact4", "fact5"]
    result = evaluate("query", "response", [obj], ["install.md"], key_facts)
    assert result["recall_score"] == "3/5"
    assert result["retrieved"] == [True, True, True, True, False]
    assert result["recalled"] == [True, True, True, False, False]
    assert result["notes"] == "missed Docker and K8s details"
    mock_retrieval.assert_called_once_with("query", key_facts, "some chunk text")
    mock_recall.assert_called_once_with("query", key_facts, "response")
