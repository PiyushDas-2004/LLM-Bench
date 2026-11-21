"""Text normalisation and lexical-overlap helpers shared by tasks and metrics."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
CITATION_RE = re.compile(r"\[(\d+)\]")


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles, squash spaces."""
    lowered = text.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", lowered).split())


def answer_tokens(text: str) -> list[str]:
    """Tokens of the normalised answer."""
    return normalize_answer(text).split()


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between normalised prediction and reference (SQuAD definition)."""
    pred, ref = answer_tokens(prediction), answer_tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists (O(len(a)*len(b)))."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure (beta = 1) over normalised tokens."""
    pred, ref = answer_tokens(prediction), answer_tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    lcs = lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains_phrase(haystack: str, needle: str) -> bool:
    """Whole-token containment of the normalised ``needle`` in the normalised ``haystack``."""
    hay, nee = normalize_answer(haystack), normalize_answer(needle)
    if not nee:
        return False
    return f" {nee} " in f" {hay} "


def extract_json(text: str) -> Any:
    """Parse JSON from model output, tolerating code fences and surrounding prose.

    Raises:
        ValueError: if no JSON value can be decoded.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            return value
    raise ValueError("no JSON value found in output")


def strip_citations(text: str) -> str:
    """Remove ``[n]`` citation markers."""
    return " ".join(CITATION_RE.sub(" ", text).split())


def citations(text: str) -> list[int]:
    """Citation numbers ``[n]`` in order of appearance (duplicates removed)."""
    seen: list[int] = []
    for match in CITATION_RE.findall(text):
        number = int(match)
        if number not in seen:
            seen.append(number)
    return seen
