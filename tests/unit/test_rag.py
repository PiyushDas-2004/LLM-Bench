from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.datasets import Sample, load_samples
from llmbench.embeddings import HashEmbedder
from llmbench.errors import ConfigError
from llmbench.prompts import PromptTemplate
from llmbench.providers import ErrorClass, MockProvider, ProviderError
from llmbench.rag import (
    BM25Retriever,
    Chunk,
    ChunkingConfig,
    Document,
    HybridRetriever,
    InMemoryVectorStore,
    LLMReranker,
    RAGPipeline,
    RAGSweepConfig,
    RerankingRetriever,
    RetrievedChunk,
    VectorRetriever,
    chunk_corpus,
    chunk_document,
    load_corpus,
    render_sweep_markdown,
    run_sweep,
    split_sentences,
)
from llmbench.rag import metrics as rm
from llmbench.rag.bm25 import BM25

DOC = Document(
    id="d.md",
    source="d.md",
    text=(
        "The pump station opened in 1972. It serves the north valley.\n\n"
        "Water is tested every Tuesday. Results are published monthly on the website.\n\n"
        + " ".join(f"word{i}" for i in range(200))
    ),
)


def test_load_corpus(repo_root: Path, tmp_path: Path) -> None:
    docs = load_corpus(repo_root / "datasets" / "rag_mini" / "corpus")
    assert len(docs) == 8
    assert all(d.text for d in docs)
    with pytest.raises(ConfigError, match="not found"):
        load_corpus(tmp_path / "missing")
    with pytest.raises(ConfigError, match="no"):
        load_corpus(tmp_path)


@pytest.mark.parametrize("strategy", ["fixed", "sentence", "recursive"])
def test_chunking_strategies(strategy: str) -> None:
    cfg = ChunkingConfig.model_validate({"strategy": strategy, "size": 120, "overlap": 30})
    chunks = chunk_document(DOC, cfg)
    assert len(chunks) > 3
    assert all(c.doc_id == "d.md" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.text) <= 160 for c in chunks)  # small tolerance for whole words/sentences
    joined = " ".join(c.text for c in chunks)
    assert "1972" in joined
    assert "word199" in joined


def test_chunking_validation_and_helpers() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(size=100, overlap=100)
    assert split_sentences("One. Two!\n\nThree?") == ["One.", "Two!", "Three?"]
    assert chunk_corpus([DOC], ChunkingConfig(size=2000, overlap=0))[0].text.startswith("The pump")
    single = Document(id="x", source="x", text="a" * 300)
    assert chunk_document(single, ChunkingConfig(strategy="recursive", size=100, overlap=0))


def test_bm25_ranks_relevant_text_first() -> None:
    index = BM25(["the pump station opened in 1972", "water is tested weekly", "unrelated words"])
    scores = index.scores("When did the pump station open?")
    assert scores.argmax() == 0
    assert BM25([]).scores("x").size == 0


async def _chunks() -> list[Chunk]:
    return chunk_document(DOC, ChunkingConfig(strategy="sentence", size=80, overlap=0))


async def test_retrievers() -> None:
    chunks = await _chunks()
    bm25 = BM25Retriever(chunks)
    top = await bm25.retrieve("pump station opened", 2)
    assert "1972" in top[0].chunk.text
    assert [r.rank for r in top] == [1, 2]
    dense = VectorRetriever(HashEmbedder())
    await dense.index(chunks)
    assert "1972" in (await dense.retrieve("pump station opened 1972", 1))[0].chunk.text
    hybrid = HybridRetriever(bm25, dense)
    fused = await hybrid.retrieve("water tested Tuesday", 3)
    assert "Tuesday" in fused[0].chunk.text
    reranked = RerankingRetriever(bm25, LLMReranker(MockProvider(), "rr"), candidates=4)
    out = await reranked.retrieve("pump station opened", 2)
    assert len(out) == 2
    assert await VectorRetriever(HashEmbedder()).retrieve("x", 3) == []


async def test_vector_store_validation() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="same length"):
        store.add([], HashEmbedder().embed_sync(["a"]))


class _Broken(MockProvider):
    async def generate(self, request):  # type: ignore[no-untyped-def]
        raise ProviderError(ErrorClass.SERVER_ERROR, "down")


async def test_reranker_survives_provider_errors() -> None:
    chunks = await _chunks()
    items = [RetrievedChunk(c, 0.0, i + 1) for i, c in enumerate(chunks[:3])]
    out = await LLMReranker(_Broken(), "rr").rerank("q", items, 2)
    assert [r.chunk.id for r in out] == [chunks[0].id, chunks[1].id]


def _sample(answerable: bool = True) -> Sample:
    evidence = (
        [{"doc_id": "d.md", "text": "The pump station opened in 1972."}] if answerable else []
    )
    return Sample(
        id="q1",
        input="When did the pump station open?",
        reference="1972" if answerable else "UNANSWERABLE",
        metadata={"answerable": answerable, "evidence": evidence},
    )


async def test_pipeline_and_metrics(repo_root: Path) -> None:
    chunks = await _chunks()
    prompt = PromptTemplate("rag", (repo_root / "prompts" / "rag_v1.j2").read_text())
    provider = MockProvider({"accuracy": 1.0, "sample_spread": 0.0})
    pipeline = RAGPipeline(BM25Retriever(chunks), provider, "gen", prompt, top_k=2)
    answer = await pipeline.answer(_sample())
    assert answer.answer.startswith("1972")
    assert answer.cited == [1]
    assert answer.cited_chunks()[0].chunk.text.startswith("The pump")
    s = _sample()
    assert rm.context_precision(answer.retrieved, s) == 0.5
    assert rm.context_recall(answer.retrieved, s) == 1.0
    assert rm.hit_at_k(answer.retrieved, s) == 1.0
    assert rm.mrr(answer.retrieved, s) == 1.0
    assert rm.ndcg_at_k(answer.retrieved, s) == 1.0
    verifier = rm.LexicalVerifier()
    assert (
        await rm.faithfulness("The pump station opened in 1972 [1].", answer.retrieved, verifier)
        == 1.0
    )
    assert await rm.faithfulness("Dragons built it in 1500.", answer.retrieved, verifier) == 0.0
    assert (
        await rm.faithfulness(
            "I cannot answer this from the provided context.", answer.retrieved, verifier
        )
        is None
    )
    assert await rm.citation_accuracy(
        "Opened in 1972 [1]. Tested on Tuesday [2]. Bogus [9].", answer.retrieved, verifier
    ) == pytest.approx(2 / 3)
    assert await rm.citation_accuracy("No citations.", answer.retrieved, verifier) is None
    assert rm.abstention_correct("1972 [1]", s) == 1.0
    unanswerable = _sample(answerable=False)
    assert (
        rm.abstention_correct("I cannot answer this from the provided context.", unanswerable)
        == 1.0
    )
    for fn in (rm.context_precision, rm.context_recall, rm.hit_at_k, rm.mrr, rm.ndcg_at_k):
        assert fn(answer.retrieved, unanswerable) is None
    assert rm.mrr([], s) == 0.0
    assert rm.ndcg_at_k([], s) is None
    assert rm.context_precision([], s) is None
    failed = await RAGPipeline(BM25Retriever(chunks), _Broken(), "gen", prompt).answer(s)
    assert failed.error is ErrorClass.SERVER_ERROR


async def test_judge_verifier() -> None:
    from llmbench.judges import LLMJudge, Rubric

    rubric = Rubric(name="r", criteria=[{"name": "c", "description": "d"}])  # type: ignore[list-item]
    judge = LLMJudge(provider=MockProvider(), provider_name="mock", model="j", rubric=rubric)
    verifier = rm.JudgeVerifier(judge)
    assert await verifier.supported("pump station opened 1972", "The pump station opened in 1972.")
    assert not await verifier.supported("dragons", "The pump station opened in 1972.")
    assert rm.chunk_supports_evidence("anything", "!!!") is False


async def test_sweep_runs_offline_on_rag_mini(repo_root: Path, tmp_path: Path) -> None:
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(
        f"""
name: test-sweep
corpus: {repo_root}/datasets/rag_mini/corpus
questions: {repo_root}/datasets/rag_mini/questions.jsonl
prompt: {repo_root}/prompts/rag_v1.j2
generator:
  id: g
  provider: mock
  params: {{accuracy: 0.95, temperature: 0.0, sample_spread: 0.3}}
n_resamples: 200
sweep:
  chunk_size: [250, 900]
  chunk_overlap: [0]
  top_k: [1, 5]
  retrieval: [bm25, hybrid]
  reranker: [none, llm]
"""
    )
    config = RAGSweepConfig.from_yaml(cfg_path)
    assert len(config.settings()) == 16
    report = await run_sweep(config)
    assert report.n_questions == 25
    assert len(report.results) == 16
    assert {"top_k", "chunk_size", "retrieval", "reranker"} <= set(report.main_effects)
    recall_by_k = report.main_effects["top_k"]["context_recall"]
    assert recall_by_k["5"] > recall_by_k["1"]
    first = report.results[0]
    assert first.metrics["abstention_correct"] is not None
    assert first.metrics["context_recall"] is not None
    assert len(first.questions) == 25
    assert "pass" in report.best
    md = render_sweep_markdown(report)
    assert "Main effects" in md
    assert "top_k" in md


def test_sweep_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        RAGSweepConfig.from_yaml(tmp_path / "missing.yaml")


async def test_sweep_rejects_unavailable_generator_and_judge(
    repo_root: Path, tmp_path: Path
) -> None:
    base = {
        "name": "x",
        "corpus": str(repo_root / "datasets/rag_mini/corpus"),
        "questions": str(repo_root / "datasets/rag_mini/questions.jsonl"),
        "prompt": str(repo_root / "prompts/rag_v1.j2"),
    }
    bad_gen = RAGSweepConfig.model_validate(
        {**base, "generator": {"id": "g", "provider": "openai", "model": "x"}}
    )
    with pytest.raises(ConfigError, match="generator"):
        await run_sweep(bad_gen)
    no_judge = RAGSweepConfig.model_validate(
        {**base, "verifier": "judge", "generator": {"id": "g", "provider": "mock"}}
    )
    with pytest.raises(ConfigError, match="judge"):
        await run_sweep(no_judge)
    bad_judge = RAGSweepConfig.model_validate(
        {
            **base,
            "verifier": "judge",
            "generator": {"id": "g", "provider": "mock"},
            "judge": {
                "provider": "openai",
                "model": "x",
                "rubric": str(repo_root / "rubrics/correctness.yaml"),
            },
        }
    )
    with pytest.raises(ConfigError, match="judge cannot run"):
        await run_sweep(bad_judge)
    missing_prompt = RAGSweepConfig.model_validate(
        {**base, "prompt": str(tmp_path / "nope.j2"), "generator": {"id": "g", "provider": "mock"}}
    )
    with pytest.raises(ConfigError, match="prompt"):
        await run_sweep(missing_prompt)


async def test_sweep_with_judge_verifier(repo_root: Path, tmp_path: Path) -> None:
    config = RAGSweepConfig.model_validate(
        {
            "name": "j",
            "corpus": str(repo_root / "datasets/rag_mini/corpus"),
            "questions": str(repo_root / "datasets/rag_mini/questions.jsonl"),
            "prompt": str(repo_root / "prompts/rag_v1.j2"),
            "generator": {"id": "g", "provider": "mock", "params": {"temperature": 0.0}},
            "verifier": "judge",
            "judge": {
                "provider": "mock",
                "model": "j",
                "rubric": str(repo_root / "rubrics/correctness.yaml"),
            },
            "n_resamples": 100,
        }
    )
    report = await run_sweep(config)
    assert report.verifier == "judge"
    assert report.results[0].metrics["faithfulness"] is not None


def test_rag_questions_have_evidence_in_corpus(repo_root: Path) -> None:
    docs = {d.id: d.text for d in load_corpus(repo_root / "datasets/rag_mini/corpus")}
    for sample in load_samples(repo_root / "datasets/rag_mini"):
        for ev in sample.metadata["evidence"]:
            assert ev["text"] in docs[ev["doc_id"]]
