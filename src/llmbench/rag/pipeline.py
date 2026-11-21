"""Reference RAG pipeline: retrieve, build a numbered context, generate a cited answer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llmbench.datasets.schema import Sample
from llmbench.prompts.template import PromptTemplate
from llmbench.providers.base import (
    ErrorClass,
    GenerationParams,
    GenerationRequest,
    Provider,
    ProviderError,
)
from llmbench.rag.retrieval import RetrievedChunk, Retriever
from llmbench.tasks.base import get_task
from llmbench.text import citations


@dataclass
class RAGAnswer:
    """Everything produced for one question."""

    sample_id: str
    answer: str
    retrieved: list[RetrievedChunk]
    cited: list[int]
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: ErrorClass | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def cited_chunks(self) -> list[RetrievedChunk]:
        """Retrieved chunks referenced by ``[n]`` citations (1-based)."""
        return [self.retrieved[n - 1] for n in self.cited if 1 <= n <= len(self.retrieved)]


def format_context(items: list[RetrievedChunk]) -> str:
    """Numbered context block: ``[1] text`` per line."""
    return "\n".join(f"[{i}] {item.chunk.text}" for i, item in enumerate(items, start=1))


class RAGPipeline:
    """Retriever + generator.

    Args:
        retriever: Any :class:`Retriever` (bring your own to benchmark an external RAG system).
        provider: Generation provider.
        model: Generation model name.
        prompt: Template receiving ``input`` and ``context``.
        top_k: Chunks placed in the context.
        params: Generation parameters.
    """

    def __init__(
        self,
        retriever: Retriever,
        provider: Provider,
        model: str,
        prompt: PromptTemplate,
        *,
        top_k: int = 4,
        params: GenerationParams | None = None,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.model = model
        self.prompt = prompt
        self.top_k = top_k
        self.params = params or GenerationParams(temperature=0.0)

    async def answer(self, sample: Sample) -> RAGAnswer:
        """Answer one question with citations."""
        start = time.perf_counter()
        retrieved = await self.retriever.retrieve(sample.input, self.top_k)
        rendered = self.prompt.render(sample, context=format_context(retrieved))
        request = GenerationRequest.from_prompt(
            self.model,
            rendered.user,
            system=rendered.system,
            params=self.params,
            metadata={"oracle": get_task("rag").oracle(sample)},
        )
        try:
            result = await self.provider.generate(request)
        except ProviderError as exc:
            return RAGAnswer(
                sample_id=sample.id,
                answer="",
                retrieved=retrieved,
                cited=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                error=exc.error_class,
            )
        return RAGAnswer(
            sample_id=sample.id,
            answer=result.text,
            retrieved=retrieved,
            cited=citations(result.text),
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            error=result.error,
        )
