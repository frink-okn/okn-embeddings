# okn-embeddings

Python app to **produce** and **search** vector embeddings of RDF knowledge graphs. The
indexing pipeline turns graph nodes into text ("textify"), embeds them (sentence-transformers
/ torch, via `core/embedding.py`), and writes self-described Parquet artifacts plus derived
sidecars (usearch ANN index, routing sketch). The search interfaces embed a query (text or an
existing node) with the same model and run similarity search against a pluggable backend:
a Qdrant collection, or the sidecar artifacts directly with no server.

## Architecture

One shared search core powers three interfaces:

- **Core** — `core/query.py::run_similarity_search` is the single entry point for search,
  driven by the `Query` / `TextFeature` / `NodeFeature` models in `core/models.py`. It embeds
  the feature (or resolves a node's stored vector via the backend) and delegates to the
  backend, returning `TimedSearchResult` of interface-agnostic `ResultRow`s.
- **Backends** — `core/backend.py`: the `SearchBackend` protocol (`search`, `survey`,
  `vector_for_iri`, `graph_facets`) with `QdrantBackend` (filtered `query_points`, batched
  survey) and `SidecarBackend` (local embed Parquet + usearch files; each file is one graph,
  so include/exclude is store selection; refuses stores with mismatched `convention_id`).
- **HTTP JSON API** — `web/routes.py`: `POST /query` accepts a `Query` JSON body and returns
  serialized rows.
- **HTMX web UI** — `web/routes.py`: `GET /` renders the form; `POST /query-view` returns an
  HTML results partial. Templates in `web/templates/`, assets in `web/static/`.
- **CLI** — `cli/main.py`: the `okn-search` Typer app. `search` (text or node/IRI query,
  `--graph`/`--exclude-graph`, `--limit`/`--offset`, `--exact`, `--show-repr`, `--json`),
  `local` (query embed Parquet files directly by path), `survey` (per-graph top-N —
  `core/explore.py`) and `list-graphs` (point counts, `--sort name|count`).

Supporting modules:

- **indexing/** — Typer app (`okn-indexing`) that produces the artifacts:
  `textify` materializes RDF (HDT) graphs into embedding-text records (`--jsonl`,
  `--manifest` for provenance; see the `textify` skill), `sample-types`/`sample-targets`
  explore the graph, `embed` turns records into a Parquet artifact (vectors + metadata
  incl. `convention_id`, the model-level comparability hash; name the graph with
  `--graph`), `build-index` builds a usearch ANN sidecar (`--dtype f16` halves it at
  ~equal recall), `eval-index` records recall in the index manifest, `build-sketch`
  builds a k-means routing sketch (certified-no bounds for cross-graph routing),
  `plot-map` renders an interactive UMAP HTML map of one or more artifacts, and
  `upload` embeds + upserts into Qdrant.
- **ann/** — sidecar artifact code: `build.py` (index build), `sidecar.py`
  (`SidecarStore`: lazy, mmap-friendly search over one artifact), `sketch.py`
  (routing sketches), `eval.py`, `plot.py`. NOTE: `ann/__init__.py` deliberately
  imports torch before usearch (bundled-libomp clash, NumKong#373).
- **evaluation/** — compares kNN vs ANN search quality and renders a markdown report.
  Dev-only dependency group (`evaluation`): pandas, umap-learn.
- **config/** — `AppSettings` (pydantic-settings) loaded from `default.env` then `.env`;
  `AppContext.from_env()` constructs the client, embedder, and search backend.

## Requirements

- Python 3.12+
- For the qdrant backend: a running Qdrant instance (default `http://127.0.0.1:6663`).
  The sidecar backend needs only artifact files.

## Configuration

Env vars (defaults in `default.env`, overridable via `.env`):

- `SEARCH_BACKEND` (`qdrant` | `sidecar`), `SIDECAR_PATHS` (glob over embed Parquet files)
- `QDRANT_LOCATION`, `QDRANT_COLLECTION` (`OKN-Graph`), `QDRANT_HNSW_EF`, `QDRANT_TIMEOUT`
- `MODEL_NAME` (`sentence-transformers/all-MiniLM-L6-v2`), `EMBED_DEVICE`
  (`auto` → CUDA > MPS > CPU), `EMBED_BATCH_SIZE`, `EMBED_THREADS` (CPU only)
- Web server: `HOST`, `PORT`, `NUM_WORKERS`, `DEBUG`, `SCRIPT_NAME` (subdir hosting, gunicorn)

Serverless demo: `SEARCH_BACKEND=sidecar SIDECAR_PATHS='data/*/records.parquet' make dev`

## Common commands

Use `uv` for everything.

- `make dev` — run the Flask dev server
- `make dev-gunicorn` — run under gunicorn
- `make test` — `uv run pytest`
- `make lint` — `ruff check` + `ruff format --check`
- `make format` — `ruff check --fix` + `ruff format`
- `make docker-build` / `make docker-run`
- `uv run okn-indexing --help` — indexing/artifact CLI
- `uv run okn-search --help` — search CLI

The artifact pipeline end-to-end:
`textify --jsonl --manifest … && embed --graph NAME … && build-index … && build-sketch …`

## Conventions

- Style enforced by ruff: line length 80, target py312, rule set B/E/F/I/S/W. Run
  `make format` before committing.
- Do not add `from __future__ import annotations`; native 3.12 typing is fine.
- All three interfaces should go through `run_similarity_search` — add features to the core
  and the `Query` model, or to the `SearchBackend` protocol, rather than duplicating search
  logic per interface. Shared helpers `build_feature`/`build_query` (`core/models.py`) and
  `summarize_point` (`core/results.py`) keep request assembly and result formatting in one
  place; `core/explore.py::run_survey` returns render-agnostic data.
- Artifacts are self-described: model identity, `convention_id`, and textify provenance live
  in the Parquet footer; derived sidecars record their parent's sha256 in their manifests.
  Design notes live in `docs/` (`local-vs-qdrant.md` benchmarks, `notes-target-filtering.md`).
