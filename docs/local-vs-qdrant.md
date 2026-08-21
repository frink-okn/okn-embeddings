# Local sidecar vs Qdrant

Rough numbers for the same 250k-record corpus (ubergraph `owl:Class`,
`all-MiniLM-L6-v2` vectors, HNSW `m=16`) served two ways:

- **Local** — `okn-search local` reading the `.parquet` + `.usearch` sidecar
  in-process.
- **Qdrant** — same records upserted into a single-node Qdrant container,
  HTTP or gRPC.

Measured on an M-series Mac, median of 50 warm reps per query, `--limit 5`.

## Search-time cost per query (ms)

| beam width | local usearch | Qdrant HTTP | Qdrant gRPC |
|---|---:|---:|---:|
| ef=64  | 0.21 | 2.33 | 1.87 |
| ef=500 | 1.51 | 4.03 | 3.99 |
| flat scan | 5.8 | – | – |

Query embedding adds ~4-5 ms on top of any of the above, same for both
backends.

## Recall (from `eval-index`, recorded in the sidecar manifest)

| ef | recall@10 | recall@100 |
|---|---:|---:|
| 64 (build-time default) | 0.987 | 0.943 |
| 256 | 0.992 | 0.977 |
| flat scan | 1.000 | 1.000 |

## Cold start (fresh `uv run okn-search …`)

~6 s either way. `uv run python -c pass` alone is ~56 ms, so the rest is
import + model load, not `uv` overhead: torch + transformers +
sentence-transformers pull in ~2 s of imports, and the model + tokenizer load
adds another ~1.5 s. A long-lived embed worker would be the biggest lever
here — search is already sub-10 ms end-to-end once warm.

## Takeaway

For single-machine, single-graph queries the sidecar is ~2.5× faster on the
search itself and simpler to ship (two files, no service). Qdrant earns its
keep with cross-graph filters, concurrency, remote access, and mutation
without rebuilds.
