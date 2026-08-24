---
name: okn-rdf-textify
description: Convert RDF nodes into textual representations for the purpose of creating text embeddings
---

Use this skill when asked to design and evaluate an indexing configuration to be used with the `okn-indexing textify` command to create text embeddings from RDF graphs.

The overall purpose is to generate textual representations of selected RDF nodes so they can be embedded with a text embedding model. RDF graphs are usually too broad and heterogeneous to embed every node directly. A good indexing configuration chooses which RDF types should become embedding documents, how those documents should be labeled, which graph edges should contribute text, and which noisy or structural graph regions should be skipped.

At a high level, the `okn-indexing` textification algorithm starts from configured RDF types, finds subject IRIs of those types, assigns each root node a human-readable label, walks selected outgoing graph edges, and turns predicate/object pairs into compact text lines. Object IRIs are represented by derived labels rather than full nested graph dumps. Label profiles can be defined separately from textification targets, so helper classes can provide good labels when they appear during graph walking without becoming standalone embedding documents. The output records group equivalent text under one record, preserve source IRIs, and include both a display label and embedding text.

Predicate selection works two ways. `ignore_predicates` is a blacklist; `include_predicates` is an allowlist that, when non-empty, restricts the walk to exactly those predicates (with `ignore_predicates` still applied within it). Choose by which list is smaller and more stable: a graph with a handful of noisy predicates wants a blacklist, while a graph whose relation vocabulary is large or open-ended wants an allowlist.

The end product is a concise, evidence-backed `config.toml` for `okn-indexing textify`, plus a report explaining the choices.

## CLI Utilities

The `okn-indexing` CLI is used to explore an HDT graph, evaluate candidate configurations, and produce final textification records. You may not have a virtual environment activated, so prefix any CLI invocation of this or other Python commands with `uv run`.

- `okn-indexing sample-types`: graph exploration. Use this before drafting a config. It scans RDF types, counts instances, samples subject IRIs, reports direct literal predicates, and reports outgoing object predicates with object type and label evidence. This helps identify good root targets, possible label predicates or label profiles, and noisy predicates to ignore.
- `okn-indexing sample-targets`: config evaluation. Use this after drafting or editing a config. It runs the configured target definitions with a small per-target limit and writes representative sample documents. This is the main tool for judging whether the config produces compact, readable, semantically useful embedding text.
- `okn-indexing textify`: final or bounded materialization. During config design, run it only with a small `--limit` to inspect output shape. Do not run the full unbounded textification/materialization process unless the user explicitly approves it after reviewing the report.

## Workflow

1. If not provided, ask for a graph to run this skill against. Currently the library only supports HDT files.
2. Create a working folder in the same directory as the graph called `${GRAPH_NAME}-textify`.
3. Run `okn-indexing sample-types` on the HDT graph to understand graph shape.
    - Use small per-type limits first, for example `--limit 2 --values-limit 2`.
    - Prefer JSON for machine review and text for human inspection.
4. Identify candidate root types.
    - Prefer entity-like types that produce compact, meaningful documents.
    - Avoid structural/helper types, extremely numerous measurement/value nodes, geometries, units, blank-node-like containers, and classes that only have ontology statements (although, of course, include ontology statements if the whole graph is an OWL ontology).
    - Record each type count and what the type appears to represent.
5. For each candidate type, inspect:
    - direct literal predicates that can become labels or useful text.
    - outgoing object predicates that lead to useful labeled entities.
    - object type distributions and object label predicates.
    - high-fanout predicates. Read `mean_per_subject` (shown as `per subject` in text output), not `count`, which is only occurrences across the sampled subjects. A predicate averaging one value per subject is an annotation; one averaging tens needs a decision.
    - High fanout by itself does not mean "ignore" — what matters is whether the values are *exchangeable*. Ask whether an arbitrary handful of them still represents the whole:
        * **Exchangeable** — siblings of the same kind, none more correct than another: the counties of a state, the authors of a paper, the ingredients of a product. Sampling a few is genuinely informative ("counties: Alameda, Butte, Calaveras"), and `predicate_limit` is exactly the right tool. Keep the predicate and cap it.
        * **Ranked** — values that differ in how informative they are, typically an inferred or transitive hierarchy. A materialized `subClassOf` closure holds one direct parent among dozens of increasingly generic ancestors, so an arbitrary subset almost always misses the useful one and yields "temporally related to: occurrent". Here `predicate_limit` cannot help and the predicate is better excluded: the values are both uninformative and drawn from a small pool of generic terms that recur across the whole corpus, so they dilute every record in the same direction.
    - Sampled values are chosen by a stable hash, so the choice is reproducible but not "the first" or "the best" ones. That is fine when the values are exchangeable and a problem when they are ranked.
    - **Also inspect the target class node's own annotations** (query the class IRI directly for its label, comment, and definition). The sampler reports *instance* predicates, so annotations on the class itself are a blind spot — yet the `type:` line renders from exactly those. A class with no `rdfs:label` falls back to its humanized IRI fragment, which may be an opaque acronym; when the class carries a descriptive `rdfs:comment` instead, adding that comment as a *trailing* per-target label predicate turns the type line into real signal (instance labels still win because they match earlier in the list). Example: a class `NSDUH` with only `rdfs:comment "Variables from the National Survey on Drug Use and Health dataset."` — the fallback renders the acronym; the comment renders the full name.
6. Draft `config.toml`.
    - Look at the file `indexing/models.py` for information about what a graph configuration consists of. You can also review `indexing/index.py` to understand the indexing algorithm.
    - Set target `type` values explicitly.
    - Add `label_predicates` that are supported by the sampled data.
    - Prefer `label_predicates` when a graph already has appropriate RDF predicates for names, titles, labels, or descriptions.
    - Only use `label_template`, `label_fields`, or `label_profiles` when no appropriate RDF label predicate exists, or when direct labels are too weak, too generic, or too verbose.
    - Do not create boilerplate label profiles that simply map `{name}` to a normal RDF name predicate; put that predicate in `label_predicates` instead.
    - Before adopting a template, check the *coverage* of every field predicate. A template whose fields are missing on most roots renders as debris (`"apoptotic process ()"`), so a template is only worth it when its fields are nearly always present.
    - Use `label_profiles` when helper classes need derived labels during graph walking without becoming textification targets.
    - Preserve `rdf:type` in embedding text by default because it is usually useful signal. Ignore type only when it is clearly unhelpful or misleading, such as ontology meta-types like `owl:Class`. In particular, when a target's `type` is the only type its roots carry, every record gets the same type line: drop it.
    - Set conservative `predicate_limit` and `expansion_limit`; prefer concise embedding text for `all-MiniLM-L6-v2`.
    - Choose between `ignore_predicates` and `include_predicates` (see above). Reach for the allowlist when the sampler shows many distinct low-value predicates, when they follow no shared prefix you could enumerate, or when the graph's vocabulary will grow — a blacklist silently lets new noise back in, while an allowlist stays correct.
    - Put the label predicate itself in neither list but expect it in the walk: the embedding text already opens with a `label:` line, so leaving `rdfs:label` walkable prints the label twice. Exclude it from the walk (it is still used for labeling, which does not go through the walk).
    - Set `expansion_limit = 0` unless you have a specific reason not to. `build_embedding_text` emits only level-0 lines, so a deeper walk is computed and then discarded — on a large graph that is a wasted subject scan per object.
7. Run `okn-indexing sample-targets` and, when useful, `okn-indexing textify` with small `--limit` values.
8. Iterate until the sampled embedding text is compact, readable, and semantically useful.
9. Write a report in the folder with the rest of the output titled `README.md`.
    - commands run.
    - included target types and type counts.
    - how many documents would be created for all targeted types in total.
    - one summary line for every targeted class describing what the class appears to represent and how many instances exist.
    - skipped types and why.
    - For each targeted type, include:
        * a summary of what this targeted class seems to represent.
        * a count of how many nodes of this type are in the graph.
        * label strategy.
        * ignored predicates and why.
        * sample embedding texts.
    - Then write:
        * a wrap-up assessment that states which representations look strongest, which look weak or noisy, which targets should be kept or reconsidered, and the highest-value next iterations.
        * remaining uncertainties or recommended next iterations.
10. Stop after creating the report. Ask the user whether to run full textification/materialization. Do not run unbounded `okn-indexing textify` automatically.

## Asking for Guidance

Prefer using sampler evidence to make routine decisions. Do not ask the user about facts the sampler can answer, such as type counts, available predicates, sample values, object type distributions, or whether a predicate is high-fanout.

Stop and ask for advice when a decision depends on domain intent, user priorities, or product semantics rather than graph structure. When asking, present a short set of concrete options and ask the user to choose. Include a recommended option when one seems best from the evidence.

Good reasons to ask:

- whether a borderline type should be a root target, context-only helper, or skipped.
- whether a high-volume class is worth embedding despite cost.
- whether users are expected to search for experimental events, measurements, publications, locations, controlled vocabulary terms, or higher-level entities.
- whether labels should prioritize identifiers, dates, human names, titles, descriptions, or compact summaries.
- whether a domain-specific predicate is semantically important or noisy when sampler evidence is ambiguous.
- whether long or numeric documents should be embedded directly or represented through parent entities.

Good question shape:

```text
For `WeatherObservation` (149,774 instances), which direction should I take?

1. Skip it as a root target and keep weather only as context. Recommended because the documents are numerous and numeric-heavy.
2. Include it as a target with aggressive predicate limits. Useful if users need direct weather-record search.
3. Defer it and document the uncertainty. Useful if we need domain input before deciding.
```

## Quality Bar

- The config should not try to embed everything.
- Each target should correspond to a thing a user may reasonably search for.
- Embedding text should be concise enough for a small sentence-transformer model.
- Label choices should be stable and human-scannable in a browsing interface.
- Every non-obvious config choice should be backed by sampler evidence.
- Every target in the config should be represented in the report by a rationale, an instance count, a short textual class summary, and at least one sample document.
- The sample-document section should be self-contained: a reader should not need to scroll back to the target table to learn what each sampled class represents.
- The report should end with an opinionated wrap-up based on the sampled documents, not just a neutral inventory.
- The skill must not run full unbounded textification/materialization on its own. Bounded `--limit` runs are allowed for evaluation; full runs require explicit user approval after the report is written.
- The algorithm should include each root document's RDF type by default. Excluding type requires a clear reason, such as avoiding unhelpful ontology/meta labels.
