"""Checked two-stage reference plans for STR and HSTR.

Only deterministic source handling, selector calls, and stable atomic packing
are implemented here. No language model, retrieval index, answer generation,
semantic temporal resolver, or benchmark QA evaluation is called. ``work`` is
sum(call_input_count ** 2), an oracle-work diagnostic, not measured runtime.

STR calls the global selector on a complete supplied domain. HSTR partitions
that same domain, calls a local selector on every block, and calls the global
selector on the retained union. Local and global selectors may differ;
agreement of outputs requires a compatibility condition beyond the checked
source/capacity contract. The fixed-score and fixed-signature selectors below
supply concrete exact subclasses when used at both stages.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Source:
    source_id: str
    score: float
    text: str = ""
    source_time: str = ""
    signature: str | None = None


Selector = Callable[[Sequence[Source], int], Iterable[Source]]
TokenCost = Callable[[Source], int]


@dataclass(frozen=True)
class FilterCall:
    stage: str
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]

    @property
    def input_count(self) -> int:
        return len(self.input_ids)

    @property
    def output_count(self) -> int:
        return len(self.output_ids)


@dataclass
class Trace:
    calls: list[FilterCall] = field(default_factory=list)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(call.input_count for call in self.calls)

    @property
    def work(self) -> int:
        return sum(size * size for size in self.sizes)

    @property
    def maximum_input(self) -> int:
        return max(self.sizes, default=0)


@dataclass(frozen=True)
class Run:
    algorithm: str
    evidence: tuple[Source, ...]
    unique_sources: int
    capacity: int
    cuts: tuple[int, ...]
    partition_mode: str
    trace: Trace

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(right - left for left, right in zip(self.cuts, self.cuts[1:]))

    @property
    def interval_size(self) -> int:
        return max(self.block_sizes, default=0)


@dataclass(frozen=True)
class LedgerEntry:
    source_id: str
    source_time: str
    token_cost: int


@dataclass(frozen=True)
class PackedEvidence:
    evidence: tuple[Source, ...]
    ledger: tuple[LedgerEntry, ...]
    skipped_ids: tuple[str, ...]
    used_tokens: int
    budget: int
    overhead_tokens: int


def _integer(value: int, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _priority(source: Source) -> tuple[float, str]:
    return (-source.score, source.source_id)


def _checked_unique(sources: Iterable[Source]) -> tuple[Source, ...]:
    """Validate immutable identities and retain the first object for each ID."""
    by_id: dict[str, Source] = {}
    for source in sources:
        if not isinstance(source, Source):
            raise ValueError("Each input must be a Source record.")
        if not isinstance(source.source_id, str) or not source.source_id:
            raise ValueError("Every source must have a nonempty string ID.")
        if (isinstance(source.score, bool)
                or not isinstance(source.score, (int, float))
                or not isfinite(source.score)):
            raise ValueError("Scores must be finite numbers.")
        if not isinstance(source.text, str) or not isinstance(source.source_time, str):
            raise ValueError("Source text and canonical timestamps must be strings.")
        if source.signature is not None and (
                not isinstance(source.signature, str) or not source.signature):
            raise ValueError("A signature must be a nonempty string or None.")
        previous = by_id.get(source.source_id)
        if previous is not None and previous != source:
            raise ValueError(f"Conflicting records for source ID {source.source_id!r}.")
        by_id.setdefault(source.source_id, source)
    return tuple(by_id.values())


def canonical_sources(sources: Iterable[Source]) -> tuple[Source, ...]:
    """Deduplicate stable IDs, then order by (canonical timestamp, source ID).

    Supply timestamps in one lexicographically sortable representation, such
    as ISO 8601 UTC. The empty default sorts before dated records. This function
    does not infer event times from text. Sorting/preparing the source domain
    is separate from Trace.work, which records only semantic selector calls.
    """
    return tuple(sorted(_checked_unique(sources),
                        key=lambda source: (source.source_time, source.source_id)))


def top_b(sources: Sequence[Source], b: int) -> tuple[Source, ...]:
    """Fixed-score top b with a global source-ID tie break."""
    _integer(b, "Frontier capacity b", minimum=1)
    return tuple(sorted(sources, key=_priority)[:b])


def canonical_witness(sources: Sequence[Source], b: int) -> tuple[Source, ...]:
    """Best source per fixed signature, then best b distinct representatives.

    Signatures are fixed before a retrieval run; None marks a source excluded
    by this example selector. Priority is always (-score, source_id), both
    within and between signatures. No signature is recomputed from the call's
    context. Used at local and global stages, this selector is composable.
    """
    _integer(b, "Frontier capacity b", minimum=1)
    representative: dict[str, Source] = {}
    for source in sources:
        if source.signature is None:
            continue
        previous = representative.get(source.signature)
        if previous is None or _priority(source) < _priority(previous):
            representative[source.signature] = source
    return tuple(sorted(representative.values(), key=_priority)[:b])


def _filter(candidates: tuple[Source, ...], b: int, selector: Selector,
            trace: Trace, stage: str) -> tuple[Source, ...]:
    """Reject invalid selections and restore original records in input order."""
    selected = tuple(selector(candidates, b))
    available = {source.source_id: source for source in candidates}
    if len(selected) > b:
        raise ValueError("Selector exceeded the frontier capacity.")
    selected_ids: set[str] = set()
    for source in selected:
        if not isinstance(source, Source) or source.source_id not in available:
            raise ValueError("Selector returned a source outside its input.")
        if source != available[source.source_id]:
            raise ValueError("Selector altered a source record.")
        if source.source_id in selected_ids:
            raise ValueError("Selector returned a repeated source ID.")
        selected_ids.add(source.source_id)
    result = tuple(source for source in candidates if source.source_id in selected_ids)
    trace.calls.append(FilterCall(stage,
                                 tuple(source.source_id for source in candidates),
                                 tuple(source.source_id for source in result)))
    return result


def ceil_cuberoot(value: int) -> int:
    """Exact nonnegative integer ceiling, including arbitrarily large inputs."""
    _integer(value, "Cube-root input", minimum=0)
    low, high = 0, 1 << ((value.bit_length() + 2) // 3)
    while low < high:
        middle = (low + high) // 2
        if middle ** 3 < value:
            low = middle + 1
        else:
            high = middle
    return low


def default_interval_size(n: int, b: int) -> int:
    """c = min(n, ceil(cuberoot(n*b*b))); return zero for an empty domain."""
    _integer(n, "Source count n", minimum=0)
    _integer(b, "Frontier capacity b", minimum=1)
    return min(n, ceil_cuberoot(n * b * b)) if n else 0


def _partition_cuts(n: int, b: int, interval_size: int | None,
                    cuts: Sequence[int] | None) -> tuple[tuple[int, ...], str]:
    if interval_size is not None and cuts is not None:
        raise ValueError("Choose interval_size or explicit cuts, not both.")
    if interval_size is not None:
        _integer(interval_size, "Interval size c", minimum=1)
    if cuts is not None:
        boundaries = tuple(cuts)
        for boundary in boundaries:
            _integer(boundary, "Partition boundary", minimum=0)
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != n:
            raise ValueError("Cuts must start at 0 and end at the unique source count.")
        if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("Cuts must strictly increase; empty blocks are not allowed.")
        if n == 0 and boundaries != (0,):
            raise ValueError("The empty domain has cuts (0,).")
        return boundaries, "explicit-cuts"
    if n == 0:
        return (0,), "count-balanced"
    c = min(n, interval_size) if interval_size is not None else default_interval_size(n, b)
    return tuple(range(0, n, c)) + (n,), "count-balanced"


def str_retrieve(sources: Iterable[Source], b: int,
                 selector: Selector = top_b) -> Run:
    """STR: one global call on every unique source in the supplied domain."""
    _integer(b, "Frontier capacity b", minimum=1)
    units = canonical_sources(sources)
    trace = Trace()
    output = _filter(units, b, selector, trace, "joint") if units else ()
    return Run("STR", output, len(units), b,
               (0, len(units)) if units else (0,), "flat", trace)


def hstr_retrieve(sources: Iterable[Source], b: int,
                  global_selector: Selector = top_b, *,
                  local_selector: Selector | None = None,
                  interval_size: int | None = None,
                  cuts: Sequence[int] | None = None) -> Run:
    """HSTR: disjoint local calls followed by a global call on their union.

    Explicit cuts refer to canonical unique-source positions. Natural temporal
    intervals can be passed as cuts=(0,1,2,5) for the paper's five-source graph.
    They need not coincide with count-balanced blocks. The default local
    selector equals the global selector, but distinct valid selectors are
    accepted. For n>0 and a nonempty retained union the global call is ALWAYS
    performed, including J=1. An empty retained union returns empty without
    calling F(empty), using the selector contract F(empty)=empty.
    """
    _integer(b, "Frontier capacity b", minimum=1)
    units = canonical_sources(sources)
    boundaries, mode = _partition_cuts(len(units), b, interval_size, cuts)
    trace = Trace()
    local = global_selector if local_selector is None else local_selector
    retained: list[Source] = []
    for j, (start, stop) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        retained.extend(_filter(units[start:stop], b, local, trace, f"local:{j}"))
    # Disjoint blocks and checked local outputs already imply distinct IDs.
    # Canonicalization makes the joint-call domain and ordering explicit.
    union = canonical_sources(retained)
    output = _filter(union, b, global_selector, trace, "joint") if union else ()
    return Run("HSTR", output, len(units), b, boundaries, mode, trace)


def pack_in_order(sources: Iterable[Source], budget: int, *,
                  token_cost: TokenCost, overhead_tokens: int = 0) -> PackedEvidence:
    """Atomic stable packing with a source ledger and a supplied exact cost.

    This function preserves the supplied priority order (it does not sort it).
    It deduplicates by first occurrence, skips an over-budget record in full,
    and still considers later shorter records. token_cost(source) must include
    that record's complete serialized text, time annotation, and ledger cost.
    overhead_tokens reserves shared query/specification overhead. No tokenizer
    or temporal normalization is supplied by this reference implementation.
    """
    _integer(budget, "Evidence budget", minimum=0)
    _integer(overhead_tokens, "Shared overhead", minimum=0)
    if overhead_tokens > budget:
        raise ValueError("Shared overhead exceeds the evidence budget.")
    units = _checked_unique(sources)
    retained: list[Source] = []
    ledger: list[LedgerEntry] = []
    skipped: list[str] = []
    used = overhead_tokens
    for source in units:
        cost = token_cost(source)
        _integer(cost, "Serialized record cost", minimum=0)
        if used + cost > budget:
            skipped.append(source.source_id)
            continue
        retained.append(source)
        ledger.append(LedgerEntry(source.source_id, source.source_time, cost))
        used += cost
    return PackedEvidence(tuple(retained), tuple(ledger), tuple(skipped),
                          used, budget, overhead_tokens)


def figure_one_sources() -> tuple[Source, ...]:
    """The exact five-source graph; global top 3 are all the same beach event."""
    return (
        Source("u1", 2, "Alice joined tree planting.", "2025-01-15", "A"),
        Source("u2", 1, "Alice volunteered at an animal shelter.", "2025-03-15", "B"),
        Source("u3", 5, "Alice joined the beach-cleanup event.", "2025-05-10", "C"),
        Source("u4", 4, "Another mention of that same beach cleanup.", "2025-05-20", "C"),
        Source("u5", 3, "A third mention of that same beach cleanup.", "2025-06-02", "C"),
    )


def selector_counterexamples() -> dict[str, dict[str, tuple[str, ...]]]:
    """Source/cap compliance alone does not imply local/global compatibility."""
    sources = tuple(Source(f"s{i}", i) for i in range(4))

    def bottom_b(items: Sequence[Source], b: int) -> tuple[Source, ...]:
        return tuple(sorted(items, key=lambda item: (item.score, item.source_id))[:b])

    def contextual(items: Sequence[Source], b: int) -> tuple[Source, ...]:
        return top_b(items, b) if len(items) > 2 else bottom_b(items, b)

    cases = {
        "incompatible_local_global": (
            str_retrieve(sources, 1, top_b),
            hstr_retrieve(sources, 1, top_b, local_selector=bottom_b, cuts=(0, 2, 4))),
        "same_contextual_selector": (
            str_retrieve(sources, 1, contextual),
            hstr_retrieve(sources, 1, contextual, cuts=(0, 2, 4))),
    }
    return {name: {run.algorithm: tuple(source.source_id for source in run.evidence)
                   for run in runs} for name, runs in cases.items()}
