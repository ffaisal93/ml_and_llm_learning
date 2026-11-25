"""Offline teaching examples for F071–F080; standard library only.

The retriever is lexical TF-IDF and the answer is an exact evidence excerpt.
No embedding model or LLM is secretly called. Agent and judge dependencies are
injected so deterministic examples and failure tests do not require a network.
"""
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
import re


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    version: int
    text: str

    @property
    def citation(self):
        return f"{self.document_id}@v{self.version}#{self.chunk_id}"


def words(text):
    # A simple teaching tokenizer, not a language-model token counter.
    return re.findall(r"\w+(?:-\w+)*", text.lower())


def retrieve(query, chunks, permissions, user, limit=3):
    # permissions is SERVER-owned current access state, never model-provided.
    allowed = [c for c in chunks if user in permissions.get(c.document_id, set())]
    if not allowed or limit < 1:
        return []
    counts = [Counter(words(c.text)) for c in allowed]
    document_frequency = Counter(term for count in counts for term in count)
    idf = {term: 1 + math.log((len(allowed) + 1) / (frequency + 1))
           for term, frequency in document_frequency.items()}

    def vector(text_counts):
        weighted = {term: count * idf[term] for term, count in text_counts.items()
                    if term in idf}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        return {term: value / norm for term, value in weighted.items()} if norm else {}

    query_vector = vector(Counter(words(query)))
    hits = []
    for chunk, count in zip(allowed, counts):
        chunk_vector = vector(count)
        score = sum(value * chunk_vector.get(term, 0) for term, value in query_vector.items())
        if score > 0:
            hits.append((score, chunk))
    return sorted(hits, key=lambda hit: (-hit[0], hit[1].citation))[:limit]


def extractive_answer(query, chunks, permissions, user):
    hits = retrieve(query, chunks, permissions, user)
    if not hits:
        return {"status": "no_evidence", "answer": None, "citations": []}
    evidence = hits[0][1]
    # Exact excerpt: this baseline does not invent an LLM-generated explanation.
    return {"status": "evidence_excerpt", "answer": evidence.text,
            "citations": [evidence.citation]}


def word_chunks(text, size=80, overlap=15):
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def reciprocal_rank_fusion(rankings, offset=60):
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        # Duplicate IDs must not occupy extra ranks or add extra votes.
        unique_ranking = list(dict.fromkeys(ranking))
        for rank, chunk_id in enumerate(unique_ranking, start=1):
            if chunk_id not in seen:
                scores[chunk_id] += 1 / (offset + rank)
                seen.add(chunk_id)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def retrieval_recall(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("score unanswerable questions separately")
    return len(set(retrieved_ids) & relevant) / len(relevant)


def pack_context(hits, max_words):
    if max_words < 0:
        raise ValueError("word budget must be nonnegative")
    selected, used, seen = [], 0, set()
    for _, chunk in hits:
        if chunk.citation in seen:
            continue
        seen.add(chunk.citation)
        length = len(chunk.text.split())
        if used + length <= max_words:
            selected.append(chunk)
            used += length
    return selected


def validate_search_call(proposal, allowed_collections):
    if not isinstance(proposal, dict) or set(proposal) != {"type", "name", "arguments"}:
        raise ValueError("expected exactly type, name and arguments")
    if proposal["type"] != "tool" or proposal["name"] != "search":
        raise ValueError("only the search tool is allowed")
    arguments = proposal["arguments"]
    if not isinstance(arguments, dict) or set(arguments) != {"query", "collection", "limit"}:
        raise ValueError("invalid search argument fields")
    if not isinstance(arguments["query"], str) or not 1 <= len(arguments["query"].strip()) <= 2000:
        raise ValueError("query must contain 1–2000 characters")
    if type(arguments["limit"]) is not int or not 1 <= arguments["limit"] <= 10:
        raise ValueError("limit must be an integer from 1 to 10")
    if not isinstance(arguments["collection"], str) or arguments["collection"] not in allowed_collections:
        raise PermissionError("collection is outside this user's permissions")
    return dict(arguments)


async def run_agent(planner, search, verify, allowed_collections,
                    max_steps=4, total_seconds=5, per_call_seconds=1):
    """Read-only demonstration. Async dependencies must cooperate with cancellation."""
    if max_steps < 1 or min(total_seconds, per_call_seconds) <= 0:
        raise ValueError("budgets must be positive")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + total_seconds
    history, completed_calls = [], set()
    failures = 0

    async def bounded(awaitable_factory):
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError("total deadline reached")
        return await asyncio.wait_for(awaitable_factory(), min(remaining, per_call_seconds))

    for _ in range(max_steps):
        try:
            proposal = await bounded(lambda: planner(list(history)))
            if isinstance(proposal, dict) and proposal.get("type") == "finish":
                answer = proposal.get("answer")
                if isinstance(answer, str) and verify(answer, history):
                    return {"status": "completed", "answer": answer, "history": history}
                return {"status": "unverified_answer", "answer": None, "history": history}
            arguments = validate_search_call(proposal, allowed_collections)
            signature = tuple(arguments[key] for key in ["query", "collection", "limit"])
            if signature in completed_calls:
                return {"status": "repeated_call", "answer": None, "history": history}
            observation = await bounded(lambda: search(**arguments))
            completed_calls.add(signature)
            history.append({"arguments": arguments, "observation": observation})
        except (ValueError, PermissionError) as error:
            # A malformed or unauthorized call never reaches the tool.
            return {"status": "invalid_action", "error": str(error), "history": history}
        except (TimeoutError, asyncio.TimeoutError):
            return {"status": "timeout", "answer": None, "history": history}
        except OSError as error:
            failures += 1
            history.append({"error": str(error)})
            if failures >= 2:
                return {"status": "tool_failure", "answer": None, "history": history}
    return {"status": "step_limit", "answer": None, "history": history}


def success_summary(trials):
    """Each result belongs to a task and says whether an external check passed."""
    grouped = defaultdict(list)
    for trial in trials:
        grouped[trial["task"]].append(bool(trial["success"]))
    if not grouped:
        raise ValueError("need at least one trial")
    task_rates = {task: sum(outcomes) / len(outcomes) for task, outcomes in grouped.items()}
    # Equal task weight even when tasks have different trial counts.
    return {"task_rates": task_rates,
            "macro_success": sum(task_rates.values()) / len(task_rates)}


async def rank_answers(answers, judge, concurrency=2, max_pairs=50,
                       timeout_seconds=1, retries=1):
    """All-pairs toy ranking; report failures instead of inventing judgments.

judge(left_text, right_text) returns 'left', 'right' or 'tie'. Each distinct
pair is scheduled once per run. Cross-run caches need model/rubric/text keys.
"""
    if concurrency < 1 or max_pairs < 0 or retries < 0 or timeout_seconds <= 0:
        raise ValueError("invalid judge budget")
    ids = sorted(answers)
    pair_count = len(ids) * (len(ids) - 1) // 2
    if pair_count > max_pairs:
        raise ValueError("all-pairs exceeds the comparison budget")
    semaphore = asyncio.Semaphore(concurrency)

    async def compare(left, right):
        for attempt in range(retries + 1):
            try:
                async with semaphore:
                    result = await asyncio.wait_for(
                        judge(answers[left], answers[right]), timeout_seconds
                    )
                if not isinstance(result, str) or result not in {"left", "right", "tie"}:
                    raise ValueError("invalid judge result")
                return left, right, result
            except (TimeoutError, asyncio.TimeoutError, OSError, ValueError):
                if attempt == retries:
                    return left, right, "failed"
        raise AssertionError("unreachable")

    comparisons = await asyncio.gather(*(compare(a, b) for a, b in combinations(ids, 2)))
    points, played, failures = dict.fromkeys(ids, 0.0), dict.fromkeys(ids, 0), []
    for left, right, result in comparisons:
        if result == "failed":
            failures.append((left, right))
            continue
        played[left] += 1
        played[right] += 1
        if result == "tie":
            points[left] += .5
            points[right] += .5
        else:
            points[left if result == "left" else right] += 1
    scores = {item: points[item] / played[item] if played[item] else None for item in ids}
    ranked = sorted(ids, key=lambda item: (scores[item] is None, -(scores[item] or 0), item))
    return {"ranking": ranked, "scores": scores,
            "comparisons": comparisons, "failures": failures}
