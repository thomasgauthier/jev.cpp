import os
import subprocess
import pytest
from utils import *

server = ServerPreset.jina_reranker_tiny()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.jina_reranker_tiny()


TEST_DOCUMENTS = [
    "A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
    "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
    "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
    "Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine."
]


def test_rerank():
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is",
        "documents": TEST_DOCUMENTS,
    })
    assert res.status_code == 200
    assert len(res.body["results"]) == 4

    most_relevant = res.body["results"][0]
    least_relevant = res.body["results"][0]
    for doc in res.body["results"]:
        if doc["relevance_score"] > most_relevant["relevance_score"]:
            most_relevant = doc
        if doc["relevance_score"] < least_relevant["relevance_score"]:
            least_relevant = doc

    assert most_relevant["relevance_score"] > least_relevant["relevance_score"]
    assert most_relevant["index"] == 2
    assert least_relevant["index"] == 3


def test_rerank_tei_format():
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is",
        "texts": TEST_DOCUMENTS,
    })
    assert res.status_code == 200
    assert len(res.body) == 4

    most_relevant = res.body[0]
    least_relevant = res.body[0]
    for doc in res.body:
        if doc["score"] > most_relevant["score"]:
            most_relevant = doc
        if doc["score"] < least_relevant["score"]:
            least_relevant = doc

    assert most_relevant["score"] > least_relevant["score"]
    assert most_relevant["index"] == 2
    assert least_relevant["index"] == 3


@pytest.mark.parametrize("documents", [
    [],
    None,
    123,
    [1, 2, 3],
])
def test_invalid_rerank_req(documents):
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is",
        "documents": documents,
    })
    assert res.status_code == 400
    assert "error" in res.body


def test_systemone_rejects_unknown_model_and_malformed_questions():
    global server
    server.start()

    valid_question = {
        "type": "choice",
        "criteria": {"payments": "Billing"},
    }
    unsupported = server.make_request("POST", "/v1/systemone", data={
        "model": "not-an-autojev-model",
        "state": "A payment issue",
        "questions": {"route": valid_question},
    })
    assert unsupported.status_code == 400
    assert unsupported.body["error"]["type"] == "invalid_request_error"

    malformed = server.make_request("POST", "/v1/systemone", data={
        "model": "autojev",
        "state": "A payment issue",
        "questions": {"route": {"type": "choice"}},
    })
    assert malformed.status_code == 400
    assert "criteria" in malformed.body["error"]["message"]


@pytest.mark.parametrize(
    "mode,embedding_enabled,rerank_enabled",
    [
        ("system_one", False, False),
        ("reranking", True, True),
        ("both", True, True),
        ("explicit_rank_embedding", True, True),
    ],
)
def test_classifier_and_embedding_route_capabilities(mode, embedding_enabled, rerank_enabled):
    server = ServerPreset.jina_reranker_tiny()
    server.server_reranking = mode in {"reranking", "both"}
    server.server_system_one = mode in {"system_one", "both"}
    server.server_embeddings = mode == "explicit_rank_embedding"
    server.pooling = "rank" if mode == "explicit_rank_embedding" else None
    server.start()

    systemone = server.make_request("POST", "/v1/systemone", data={
        "model": "autojev",
        "state": "A payment issue",
        "questions": {"route": {"type": "choice", "criteria": {"payments": "Billing"}}},
    })
    assert systemone.status_code == 501
    assert "autojev.format_version" in systemone.body["error"]["message"]
    if not embedding_enabled:
        for path in ("/embedding", "/embeddings", "/v1/embeddings"):
            assert server.make_request("POST", path, data={"input": "test"}).status_code == 404

    if not rerank_enabled:
        for path in ("/rerank", "/reranking", "/v1/rerank", "/v1/reranking"):
            assert server.make_request(
                "POST", path, data={"query": "test", "documents": ["test"]}
            ).status_code == 404

    if embedding_enabled:
        embedding = server.make_request("POST", "/v1/embeddings", data={"input": "test"})
        assert embedding.status_code == 200
    if rerank_enabled:
        rerank = server.make_request(
            "POST", "/rerank", data={"query": "test", "documents": ["test"]}
        )
        assert rerank.status_code == 200


@pytest.mark.parametrize(
    "query,doc1,doc2,n_tokens",
    [
        ("Machine learning is", "A machine", "Learning is", 19),
        ("Which city?", "Machine learning is ", "Paris, capitale de la", 26),
    ]
)
def test_rerank_usage(query, doc1, doc2, n_tokens):
    global server
    server.start()

    res = server.make_request("POST", "/rerank", data={
        "query": query,
        "documents": [
            doc1,
            doc2,
        ]
    })
    assert res.status_code == 200
    assert res.body['usage']['prompt_tokens'] == res.body['usage']['total_tokens']
    assert res.body['usage']['prompt_tokens'] == n_tokens


@pytest.mark.parametrize("top_n,expected_len", [
    (None, len(TEST_DOCUMENTS)),  # no top_n parameter
    (2, 2),
    (4, 4),
    (99, len(TEST_DOCUMENTS)),    # higher than available docs
])
def test_rerank_top_n(top_n, expected_len):
    global server
    server.start()
    data = {
        "query": "Machine learning is",
        "documents": TEST_DOCUMENTS,
    }
    if top_n is not None:
        data["top_n"] = top_n

    res = server.make_request("POST", "/rerank", data=data)
    assert res.status_code == 200
    assert len(res.body["results"]) == expected_len


@pytest.mark.parametrize("top_n,expected_len", [
    (None, len(TEST_DOCUMENTS)),  # no top_n parameter
    (2, 2),
    (4, 4),
    (99, len(TEST_DOCUMENTS)),    # higher than available docs
])
def test_rerank_tei_top_n(top_n, expected_len):
    global server
    server.start()
    data = {
        "query": "Machine learning is",
        "texts": TEST_DOCUMENTS,
    }
    if top_n is not None:
        data["top_n"] = top_n

    res = server.make_request("POST", "/rerank", data=data)
    assert res.status_code == 200
    assert len(res.body) == expected_len


def test_system_one_pooling_is_order_independent():
    server_binary = os.environ["LLAMA_SERVER_BIN_PATH"]
    for arguments in (["--system-one", "--pooling", "rank"], ["--pooling", "rank", "--system-one"]):
        result = subprocess.run(
            [server_binary, *arguments, "--help"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    for arguments in (["--system-one", "--pooling", "mean"], ["--pooling", "mean", "--system-one"]):
        result = subprocess.run(
            [server_binary, *arguments, "--help"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 1
        assert "--system-one requires rank pooling" in result.stderr
