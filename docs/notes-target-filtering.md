# Note: finer-grained target filtering for textify

Selecting textify roots by `rdf:type` alone is too coarse for a graph like
ubergraph, where one type (`owl:Class`) covers everything from rich disease
terms to unidentified sequencing isolates. Evidence from the ubergraph
corpus (3.88M `owl:Class` records):

- **1.31M records (34%) are provisional taxa** — labels like
  "Aspergillus sp. 6/M/2/5/3": an unidentified isolate given a species-level
  NCBI taxid on sequence deposit. Label-only documents, no synonyms or
  definitions, `referenceCount 1`. Nobody searches for them.
- ~38k more are "uncultured …" / "unidentified …" / metagenome entries.
- Strains (`has_rank NCBITaxon_strain`) are only ~47k — the visible
  irritation, but not the mass.

## Proposed mechanism

Optional per-target filters, applied to each candidate root:

```toml
[targets.class]
type = "http://www.w3.org/2002/07/owl#Class"
exclude_when = [
  { predicate = "http://purl.obolibrary.org/obo/ncbitaxon#has_rank",
    object = "http://purl.obolibrary.org/obo/NCBITaxon_strain" },
]
exclude_label_pattern = ' sp\. '
```

- `exclude_when` / `require_when`: predicate–object existence tests. With
  HDT each is a single `(iri, P, O)` probe — effectively free per root.
- `exclude_label_pattern`: a regex over the resolved label, because the
  provisional-taxon case has no clean predicate marker (they are
  `has_rank species` like real species; only the name gives them away).

## Cautions

- `biolink:category` looks like a selection axis but is **closure-polluted**
  in reasoner-materialized graphs: a bacterial strain carries
  `category: AnatomicalEntity` (2.52M nodes claim AnatomicalEntity).
  Filter on direct, single-valued predicates (`has_rank`, `inferred`),
  not materialized ones.
- Dropping strains/provisional taxa is mostly a **result-quality** play
  (label-only docs crowding real species in results) and a 1/3 corpus-size
  cut for ubergraph; measure result quality before/after when implemented.
- A cleaner marker than the label regex for provisional taxa: NCBI parks
  them under container nodes named "unclassified <Genus>" /
  "environmental samples" (e.g. NCBITaxon_1257612's direct parent is
  "unclassified Aspergillus"). A one-hop structural test — direct parent's
  label starts with "unclassified" — would be more precise, at the cost of
  needing a parent-label lookup rather than a self-contained probe.

Status: deliberately not implemented (2026-08) — probing every graph is
cheap at current corpus sizes, so this waits for evidence that junk
documents actually hurt result quality.
