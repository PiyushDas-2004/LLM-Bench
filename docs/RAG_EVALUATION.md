# RAG evaluation

`src/llmbench/rag/` evaluates retrieval-augmented generation in two layers: retrieval
quality against gold evidence passages, and generation quality (groundedness, citations,
abstention, answer correctness). It ships a reference pipeline (ingestion, chunking, BM25,
vector and hybrid retrieval, reranking, cited generation) and a sweep runner that
evaluates a grid of settings. You can also plug in an existing RAG system. General metric
and statistics definitions are in [METHODOLOGY.md](METHODOLOGY.md).

## 1. Plugging in an external RAG system

Every retriever implements one protocol (`rag/retrieval.py`):

```python
class Retriever(Protocol):
    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...
```

`RetrievedChunk(chunk, score, rank)` wraps a `Chunk(id, doc_id, text, index)`; `rank` is
1-based. Retrieval metrics use only `chunk.text` and the order; `score` is reported but not
used.

**Option A: external retriever, LLMBench generation.** Wrap your search service and pass
it to `RAGPipeline`, which builds the numbered context, calls the generator and parses
citations:

```python
import asyncio
from pathlib import Path

from llmbench.datasets import load_samples
from llmbench.embeddings import HashEmbedder
from llmbench.prompts import PromptTemplate
from llmbench.providers import MockProvider
from llmbench.rag import Chunk, RAGPipeline, RetrievedChunk
from llmbench.rag.metrics import LexicalVerifier
from llmbench.rag.sweep import score_answer


class MySearchRetriever:
    """Adapter around an external search client exposing `await search(query, limit)`."""

    def __init__(self, client) -> None:
        self.client = client

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        hits = await self.client.search(query, limit=k)
        return [
            RetrievedChunk(
                chunk=Chunk(id=h["id"], doc_id=h["doc_id"], text=h["text"], index=h.get("index", 0)),
                score=float(h["score"]),
                rank=rank,
            )
            for rank, h in enumerate(hits, start=1)
        ]


async def main(client) -> None:
    samples = load_samples("datasets/rag_mini/questions.jsonl")
    prompt = PromptTemplate("rag", Path("prompts/rag_v1.j2").read_text(encoding="utf-8"))
    pipeline = RAGPipeline(
        MySearchRetriever(client), MockProvider({"accuracy": 0.9}), "mock-rag", prompt, top_k=4
    )
    verifier, embedder = LexicalVerifier(), HashEmbedder()
    for sample in samples:
        answer = await pipeline.answer(sample)
        print(sample.id, await score_answer(answer, sample, verifier, embedder))

# asyncio.run(main(my_client))
```

Replace `MockProvider` with any provider built by `llmbench.providers.registry.create_provider`
for a real generator.

**Option B: score a complete external system.** If your system generates its own answers,
build a `RAGAnswer` from its output and the chunks it used, and call `score_answer`
directly:

```python
from llmbench.rag.pipeline import RAGAnswer
from llmbench.text import citations

text = my_system_answer  # must cite context passages as [1], [2], ... (1-based, in `retrieved` order)
answer = RAGAnswer(sample_id=sample.id, answer=text, retrieved=retrieved,
                   cited=citations(text), latency_ms=elapsed_ms)
scores = await score_answer(answer, sample, LexicalVerifier(), HashEmbedder())
```

Both examples were run against the bundled dataset with a stub client. Citation accuracy
assumes `[n]` refers to the n-th element of `retrieved`.

## 2. Ingestion and chunking

**Ingestion** (`documents.py`). `load_corpus(root)` loads every `*.md`, `*.txt` and `*.pdf`
under `root` recursively, sorted by path; a document's `id` is its path relative to the
root. PDFs need the `pdf` extra (`pypdf`, pages joined with blank lines). An empty corpus is
a configuration error.

**Chunking** (`chunking.py`). `ChunkingConfig(strategy, size, overlap)`: sizes in
characters, `size >= 50`, `0 <= overlap < size`, defaults `recursive`/500/50. Chunk ids
are `"{doc_id}#{i}"`. All strategies collapse whitespace in their output.

- **`fixed`**: splits on whitespace and greedily packs words while the running length
  (word length + 1 each) stays within `size`; a single word longer than `size` becomes its
  own chunk. The next chunk starts by stepping back over trailing words whose running
  length fits in `overlap`, but never back to the previous chunk's first word, so it always
  advances. Paragraph structure is lost.
- **`sentence`**: `split_sentences` splits paragraphs on blank lines, then sentences at
  whitespace that follows `.`, `!` or `?` and precedes an uppercase letter, digit, quote or
  parenthesis (a regex heuristic: abbreviations such as "e.g. The" split wrongly).
  Sentences longer than `size` are cut with `fixed` (no overlap). Sentences are packed
  while the joined length is at most `size`; on overflow the chunk is emitted and trailing
  whole sentences whose joined length is at most `overlap` are carried into the next chunk.
  A chunk can exceed `size` by up to the carried overlap.
- **`recursive`** (default): text of at most `size` characters is one chunk. Otherwise it
  splits on the first separator in `"\n\n"`, `"\n"`, `". "`, `" "` that produces more than
  one piece and merges pieces greedily while the joined text fits. A piece longer than
  `size` is split recursively with the next separator; when separators run out it falls
  back to `fixed`. When a new chunk starts after a flush, it is prefixed with the last
  `overlap` **characters** of the previous chunk (this can cut mid-word, and the chunk can
  exceed `size` by about `overlap`).

## 3. Retrieval

**BM25** (`bm25.py`). Tokens are lowercase `[a-z0-9]+` runs minus 33 English stop words.
With $N$ chunks, document frequency $\mathrm{df}(t)$, term frequency $f(t, d)$, chunk length
$|d|$ in tokens and mean length $\overline{|d|}$:

$$
\mathrm{idf}(t) = \ln\!\left(1 + \frac{N - \mathrm{df}(t) + 0.5}{\mathrm{df}(t) + 0.5}\right),
\qquad
\mathrm{score}(q, d) = \sum_{t \in \mathrm{set}(q)} \mathrm{idf}(t)\,
\frac{f(t,d)\,(k_1 + 1)}{f(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\overline{|d|}}\right)}
$$

with $k_1 = 1.5$, $b = 0.75$. The "+1" form keeps idf positive. Repeated query terms
count once. `BM25Retriever` returns the top $k$ by score (stable sort, so ties keep corpus
order), including zero-score chunks when fewer than $k$ match.

**Vector stores** (`stores.py`). The `VectorStore` protocol has `add(chunks, vectors)` and
`search(vector, k)`; vectors are unit-norm, so cosine is a dot product.

- `InMemoryVectorStore`: exact numpy search (stable sort), the default.
- `PgVectorStore(dsn, dim, table)`: Postgres with pgvector (`pgvector` extra). It creates the
  extension and a table `(id, doc_id, idx, body, embedding vector(dim))`, upserts by chunk
  id, and searches with `ORDER BY embedding <=> query LIMIT k`, reporting
  $1 - \text{cosine distance}$. No ANN index is created, so search is an exact scan. `dim`
  must match the embedder (`HashEmbedder` defaults to 512). Start a local database with
  `docker compose --profile postgres up -d postgres`; the live round-trip test is
  `tests/live/test_live_pgvector.py`, run with `LLMBENCH_PGVECTOR_URL` set and
  `pytest -m live -o addopts=""`. The compose file's local-development default password is
  `llmbench` (override with `POSTGRES_PASSWORD`), matching the example DSN. The
  sweep YAML always uses the in-memory store; use pgvector programmatically via
  `VectorRetriever(embedder, store=PgVectorStore(dsn, dim))`.

`VectorRetriever(embedder, store)` embeds chunks in `index()` and queries in `retrieve()`.
With the default `HashEmbedder`, "dense" retrieval is a hashed lexical match (see
METHODOLOGY.md, section 2.3); use `embedder: st:<model>` for semantic retrieval.

**Hybrid retrieval with reciprocal rank fusion.** `HybridRetriever(lexical, dense,
candidates=20)` takes the top $\max(k, 20)$ from each retriever and fuses ranks:

$$
\mathrm{RRF}(c) = \sum_{r \in \{\text{lexical}, \text{dense}\}} \frac{1}{60 + \mathrm{rank}_r(c)} ,
$$

summing only over retrievers that returned $c$ (`RRF_K = 60`). Ties are broken by chunk id.
RRF uses ranks only, so the two retrievers' score scales need not be comparable.

**Rerankers.** `RerankingRetriever(base, reranker, candidates=10)` fetches $\max(k, 10)$
candidates and lets the reranker keep $k$. `LLMReranker(provider, model)` makes one call per
candidate at temperature 0 asking for JSON `{"score": 0-10}`; a provider or parse error
scores $-1$; ties keep the original rank. `CrossEncoderReranker` uses a
`sentence-transformers` cross-encoder (optional extra). In the sweep, the LLM reranker uses
the generator's provider and model. With the mock provider the reranker's score is a
token-overlap proxy, so mock results say nothing about real LLM reranking.

## 4. Pipeline and citation format

`RAGPipeline(retriever, provider, model, prompt, top_k=4, params=temperature 0)`
(`pipeline.py`): retrieve `top_k` chunks for `sample.input`, format them as one line per
chunk, `[1] text`, `[2] text`, ..., render the prompt with `input` and `context`, generate,
and parse citations. `citations()` returns the distinct `[n]` numbers in order of first
appearance; `cited_chunks()` maps them to retrieved chunks (1-based) and ignores
out-of-range numbers. A provider error yields an empty answer with the error class.
`latency_ms` covers retrieval plus generation.

`prompts/rag_v1.j2` tells the model to use only the numbered passages, cite them as `[n]`
after the answer, and otherwise reply exactly "I cannot answer this from the provided
context.", a sentence the abstention heuristic recognises ("cannot answer").

## 5. Metrics (`rag/metrics.py`, `rag/sweep.py::score_answer`)

### 5.1 Retrieval metrics against gold evidence

Questions carry `metadata.evidence = [{doc_id, text}, ...]`. A chunk **supports** an
evidence passage if the normalised passage (METHODOLOGY.md, 2.1) is a substring of the
normalised chunk, or if at least 60% of the passage's distinct content tokens (BM25
tokens) occur in the chunk:

$$
\frac{|T(\text{evidence}) \cap T(\text{chunk})|}{|T(\text{evidence})|} \ge 0.6 .
$$

Matching is text-only; `doc_id` is not used. The coverage rule lets a passage count even
when a chunk boundary cuts it, but a chunk that keeps fewer than 60% of the passage's
content tokens does not count, so small chunk sizes are partly penalised by construction. A
chunk is *relevant* if it supports any passage. With $k$ retrieved chunks, relevance
vector $r_1..r_k$ and $E$ evidence passages:

| metric | definition |
|---|---|
| `context_precision` | $\sum_i r_i / k$ |
| `context_recall` | passages supported by at least one retrieved chunk / $E$ |
| `hit_at_k` | $\mathbb{1}[\exists i: r_i]$ |
| `mrr` | $1/\min\{i : r_i\}$, 0 if none |
| `ndcg_at_k` | $\mathrm{DCG}/\mathrm{IDCG}$ with $\mathrm{DCG} = \sum_i r_i/\log_2(i + 1)$ and IDCG placing $\min(k, \max(E, R))$ relevant chunks first, where $R = \sum_i r_i$ |

All are `None` for questions without evidence (unanswerable questions), so they average
over answerable questions only. Several overlapping chunks can support the same passage;
because the ideal count uses $\max(E, R)$, a ranking with $R$ relevant chunks on top scores
exactly 1 and nDCG stays in $[0, 1]$. (An earlier version used $\min(k, E)$ and could
exceed 1 in this case; it was fixed after this very sweep exposed it.)

### 5.2 Generation metrics

**Claims.** `answer_claims` splits the answer into sentences with `split_sentences`, strips
citations and keeps sentences with at least one content token; an abstention has no claims.
Sentence-level claims are a coarse stand-in for atomic-fact decomposition: a compound
sentence is judged as a whole.

**Verifiers** (`ClaimVerifier.supported(claim, context)`):

- `LexicalVerifier(threshold=0.7)`: supported if at least 70% of the claim's distinct
  content tokens appear anywhere in the context (a claim with no content tokens is
  supported). It is an offline proxy for entailment with known blind spots. It is a bag of
  words over all retrieved chunks combined, so tokens may come from unrelated chunks. It
  misses paraphrase. It tolerates contradictions: with $n$ supported tokens, adding one
  unsupported token (for example "not") keeps coverage $n/(n+1) \ge 0.7$ for $n \ge 3$, and
  replacing one token (for example a wrong date) keeps $(n-1)/n \ge 0.7$ for $n \ge 4$.
- `JudgeVerifier(judge)`: asks an `LLMJudge` pointwise "Is this claim fully supported by
  the reference context?" with the claim as candidate and the context as reference;
  supported iff the verdict passes (mean score $\ge$ the rubric's `pass_threshold`). Judge
  failures count as unsupported. The rubric is whatever the sweep's `judge.rubric` names;
  the bundled `correctness` rubric is written for answer correctness, so a rubric about
  support is more appropriate.

**Metrics per question** (`score_answer`):

| metric | definition |
|---|---|
| `faithfulness` | supported claims / claims, context = all retrieved chunks; `None` with no claims |
| `hallucination` | $\mathbb{1}[\text{faithfulness} < 1]$ (any unsupported claim); `None` with no claims; lower is better |
| `answer_relevance` | cosine of embeddings of question and citation-stripped answer; `None` for empty or abstaining answers |
| `citation_accuracy` | over every `[n]` in every sentence: share where $1 \le n \le k$ and passage $n$ supports that sentence; `None` without citations |
| `abstention_correct` | $\mathbb{1}[\text{abstained} \ne \text{answerable}]$ |
| `answer_f1` | token F1 of the citation-stripped answer vs the reference; answerable questions only |
| `pass` | `rag` task pass rule: abstain on unanswerable; otherwise no abstention and the `qa` criterion |

With the default hash embedder `answer_relevance` measures word overlap between question and
answer. Short factual answers ("12 April 1958") share few tokens with the question, so values
near zero are expected and not meaningful. An empty answer from a failed generation is not
an abstention, so it counts as abstention-correct on answerable questions while failing
`pass`.

Only `abstention_correct` is registered as a regular metric (usable in `llmbench run` with
`task: rag`, whose defaults are `f1` and `abstention_correct`). The retrieval and
groundedness metrics need the retrieved chunks and are available through `score_answer` and
the sweep.

## 6. Sweeps and main effects (`sweep.py`)

```bash
llmbench rag sweep examples/rag_sweep.yaml --output reports/rag_sweep.md [--json results.json] [--top 10]
```

The YAML (`RAGSweepConfig`) names the corpus, questions, prompt, a `generator` model spec,
`verifier: lexical | judge` (with a `judge` section for the latter), `n_resamples` (default
1000), `seed`, and a `sweep` grid over `chunk_strategy`, `chunk_size`, `chunk_overlap`,
`top_k`, `retrieval` (`bm25`/`vector`/`hybrid`), `reranker` (`none`/`llm`) and `embedder`.
The sweep evaluates the Cartesian product, skipping combinations with
`overlap >= size`. `examples/rag_sweep.yaml` has
$2 \times 2 \times 2 \times 2 \times 3 \times 2 \times 1 = 96$ settings on 25 questions, so
$96 \times 25 = 2400$ generations, plus $48 \times 25 \times 10 = 12{,}000$ reranker calls for
the 48 settings with the LLM reranker.

For each setting: chunk the corpus, build the retriever stack, answer every question once,
score it, and report each metric's mean with a 95% percentile bootstrap CI over questions
(seed `seed + setting index`). **Main effects**: for every factor with at least two levels,
the unweighted mean of the setting estimates at each level (marginal means over all other
factors), plus the spread (max minus min). **Best setting per metric**: the highest point
estimate (lowest for `hallucination`).

The Markdown report contains: a header (generator, verifier, question and setting counts);
the top settings sorted by `pass` with `pass`, `answer_f1`, `context_recall`, `mrr`,
`ndcg_at_k`, `faithfulness`, `abstention_correct` and their CIs; one main-effects table per
factor; the best setting per metric. `--json` adds every setting's per-question answers,
retrieved chunk ids and metric values.

**Illustrative output** (mock generator + lexical verifier, illustrative; run on
2026-11-21 with `examples/rag_sweep.yaml`). The `top_k` main effect shows the expected
precision/recall trade-off: `context_precision` 0.403 at $k = 2$ vs 0.216 at $k = 5$,
`context_recall` 0.648 vs 0.821. `answer_relevance`
averaged about 0.09 at every factor level, a property of the hash embedder rather than of
the answers. The top setting's `pass` CI was [0.88, 1.00] on 25 questions. These numbers
describe the mock, which answers from context by checking whether the first 60 characters
of an evidence passage appear in a context block; they say nothing about real models.

**Reading a sweep.** CIs are per setting, not comparison tests. Two settings with
overlapping CIs are not shown to differ, and picking the best of 96 point estimates favours
settings that got lucky on 25 questions (the winner's curse). Main effects average over
interactions (for example, a reranker that helps BM25 but hurts vector retrieval), and they
are only balanced when no combinations were skipped. Each question is answered once,
usually at temperature 0, so generation non-determinism is not measured. To confirm a
choice, compare the finalists on more questions with repeated trials using the paired tests
in METHODOLOGY.md.

## 7. The bundled `rag_mini` dataset

`datasets/rag_mini/corpus/` holds 8 Markdown documents (about 2.5-2.7 kB each, about 21 kB
in total) about a fictional utility, the Lumen Valley Water Cooperative: `billing`,
`conservation`, `governance`, `history`, `membership`, `outages`, `service_area`,
`water_quality`. `questions.jsonl` has 25 questions: 20 answerable (16 with one evidence
passage, 2 with two, 1 with three, 1 with four) and 5 unanswerable (`answerable: false`,
empty evidence, reference `UNANSWERABLE`). Difficulty tags: 10 easy, 9 medium, 6 hard.
Record format:

```json
{"id": "rag-001", "input": "When was the Lumen Valley Water Cooperative founded?",
 "reference": "12 April 1958", "tags": ["rag", "answerable", "history"], "difficulty": "easy",
 "metadata": {"answerable": true, "evidence": [{"doc_id": "history.md", "text": "The cooperative was founded on 12 April 1958 by ..."}]}}
```

The corpus is fictional so that answers cannot come from a model's pretraining data. It is
small: retrieval metrics on 20 answerable questions and abstention on 5 unanswerable ones
have wide intervals.

## 8. Limitations

- **Lexical defaults.** The hash embedder, the lexical verifier and the 0.6 evidence rule
  measure word overlap. Paraphrased evidence, paraphrased answers and contradictions that
  reuse the context's words are misjudged. Use a sentence-embedding model and
  `verifier: judge` (with a calibrated judge) for real evaluations.
- **Gold evidence quality.** Retrieval metrics are only as good as the annotated passages.
  A chunk containing equally valid evidence that was not annotated counts as irrelevant.
- **Coarse claims.** Sentences are not atomic facts; one unsupported clause makes the whole
  sentence unsupported, and `hallucination` flags any answer with one unsupported sentence.
- **Heuristic abstention.** Abstention detection is phrase matching (METHODOLOGY.md, 2.8).
  A model that abstains in other words is scored as answering.
- **Sweep statistics.** One answer per question, per-setting CIs only, no multiplicity
  control when choosing a best setting, main effects ignore interactions.
- **Mock generator.** It reads the evidence through the oracle. Mock sweeps validate the
  plumbing and show that retrieval quality propagates to answers, not how any model behaves.
- **Scale.** The in-memory store is exact and fine for small corpora; pgvector as configured
  here does an exact scan without an ANN index.
