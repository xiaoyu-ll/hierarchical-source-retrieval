# Hierarchical Source Retrieval (HSTR)

This repository contains the Python reference implementation of STR and HSTR for selecting a bounded set of evidence sources from a supplied candidate set. STR applies a selector once to the full set. HSTR partitions the sources into contiguous blocks in timestamp order, selects up to `b` sources per block, and applies a global selector to the retained union.

## Quick start

Requires **Python 3.10 or later**. No third-party packages are needed.

```sh
git clone https://github.com/xiaoyu-ll/hierarchical-source-retrieval.git
cd hierarchical-source-retrieval
```

Run the following in Python from the repository directory. The bundled example contains five sources, three of which describe the same event. `canonical_witness` retains the highest-scoring source per signature before selecting the best `b` representatives.

```python
from hierarchical_retrieval import (
    canonical_witness,
    figure_one_sources,
    hstr_retrieve,
    str_retrieve,
)

sources = figure_one_sources()
flat = str_retrieve(sources, b=3, selector=canonical_witness)
hierarchical = hstr_retrieve(
    sources, b=3, global_selector=canonical_witness, cuts=(0, 1, 2, 5)
)

for run in (flat, hierarchical):
    print(run.algorithm, [source.source_id for source in run.evidence], run.trace.work)
```

```text
STR ['u1', 'u2', 'u3'] 25
HSTR ['u1', 'u2', 'u3'] 20
```

Both plans recover the same evidence in this example. The last column is the sum of squared selector input sizes, a measure of modeled work rather than elapsed time.

## Using your own sources

Create `Source` records with stable IDs and scores. Use a consistent sortable format for `source_time`, such as ISO 8601 in UTC. For `canonical_witness`, assign a fixed `signature` to each source you want the selector to consider. The default selector, `top_b`, ranks individual sources by score.

HSTR chooses a block size by default. Pass `interval_size` to set the maximum number of sources per block, or `cuts` to specify boundaries by position after deduplication and sorting. Both retrieval functions return a `Run` with the selected `evidence` and a `trace` of selector calls.

The built-in selectors preserve the STR frontier when the same selector is used at both stages of HSTR. For custom selectors, exact recovery depends on how the local and global selectors compose: applying the global selector to the retained union must yield the same frontier as applying it to the full set.
