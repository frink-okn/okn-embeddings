from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from typing import Literal as TypingLiteral

from rdflib import BNode, Graph, Literal, Node, URIRef
from rdflib.namespace import RDF
from rdflib.util import from_n3
from rdflib_hdt import HDTStore


def load_graph(hdt_file: Path):
    store = HDTStore(str(hdt_file))
    return Graph(store=store)


# A GraphReader reads triples out of a graph as lightweight `GraphTerm`s. The
# HDT reader reads the native HDT document directly, which avoids rdflib's
# per-term object construction -- the dominant cost when materializing. This
# layer only reads; turning nodes into text is the Textifier's job.


@dataclass(frozen=True)
class GraphTerm:
    """An RDF term as a `kind` plus its string value.

    Used instead of full rdflib `URIRef`/`Literal` objects on the materialize
    hot path: constructing/validating rdflib terms for every triple is a large
    share of the cost, and we only ever need the string value here.
    """

    kind: TypingLiteral["iri", "literal", "bnode"]
    value: str

    def __str__(self) -> str:
        return self.value


class GraphReader:
    """Read access to a graph in terms of `GraphTerm`s."""

    def root_iris(self, root_type: str) -> Iterable[str]:
        raise NotImplementedError

    def root_count(self, root_type: str) -> int | None:
        """Fast subject count for `? rdf:type <root_type>`, or None if unknown.

        Used by the textify progress bar to set tqdm's `total`. A None result
        means the caller should show a countless spinner instead.
        """
        return None

    def predicate_objects(
        self, subject: Any
    ) -> Iterable[tuple[GraphTerm, GraphTerm]]:
        raise NotImplementedError

    def objects(self, subject: Any, predicate_iri: str) -> Iterable[GraphTerm]:
        raise NotImplementedError

    def term(self, node: Any) -> GraphTerm:
        raise NotImplementedError

    def types(self, node: Any) -> Iterable[str]:
        for obj in self.objects(node, str(RDF.type)):
            if obj.kind == "iri":
                yield obj.value

    def first_literal(
        self, subject: Any, predicates: Iterable[str]
    ) -> str | None:
        for predicate in predicates:
            for obj in self.objects(subject, predicate):
                if obj.kind == "literal":
                    return obj.value
        return None


class RDFLibGraphReader(GraphReader):
    """Reader over any in-memory rdflib graph (used by tests)."""

    def __init__(self, graph: Graph):
        self.graph = graph

    def root_iris(self, root_type: str) -> Iterable[str]:
        for node in self.graph.subjects(RDF.type, URIRef(root_type)):
            if isinstance(node, URIRef):
                yield str(node)

    def predicate_objects(
        self, subject: Any
    ) -> Iterable[tuple[GraphTerm, GraphTerm]]:
        node = self._to_node(subject)
        if isinstance(node, Literal):
            return
        for pred, obj in self.graph.predicate_objects(node):
            if isinstance(pred, URIRef):
                yield self.term(pred), self.term(obj)

    def objects(self, subject: Any, predicate_iri: str) -> Iterable[GraphTerm]:
        node = self._to_node(subject)
        if isinstance(node, Literal):
            return
        for obj in self.graph.objects(node, URIRef(predicate_iri)):
            yield self.term(obj)

    def term(self, node: Any) -> GraphTerm:
        if isinstance(node, GraphTerm):
            return node
        if isinstance(node, Literal):
            return GraphTerm("literal", str(node))
        if isinstance(node, BNode):
            return GraphTerm("bnode", str(node))
        return GraphTerm("iri", str(node))

    def _to_node(self, node: Any) -> Node:
        if isinstance(node, GraphTerm):
            if node.kind == "literal":
                return Literal(node.value)
            if node.kind == "bnode":
                return BNode(node.value)
            return URIRef(node.value)
        return node


class HDTGraphReader(GraphReader):
    """Reader that reads the native HDT document, skipping rdflib terms.

    Every `search_triples` call here discards the second element of the
    returned pair. Do not start using it as a triple count: for
    predicate-only patterns it is the number of distinct *subjects*, and
    `TripleIterator.size_hint()` reports that as accurate. Measured on
    Ubergraph, `? rdfs:subClassOf ?` reports 3,886,036 against 112,020,318
    triples actually iterated. Counting means iterating.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.document = graph.store.hdt_document

    def root_iris(self, root_type: str) -> Iterable[str]:
        triples, _ = self.document.search_triples("", str(RDF.type), root_type)
        for subject, _, _ in triples:
            if not subject.startswith("_:"):
                yield subject

    def root_count(self, root_type: str) -> int | None:
        # HDT's cardinality is exact for two-fixed-position patterns (the
        # docstring's warning about overcounting applies to predicate-only
        # `? P ?` patterns, which take a different index path). The subject
        # count matches the triple count here because rdf:type is unique per
        # (subject, class) pair. Blank-node subjects, filtered in root_iris,
        # are rare enough to leave the count as an upper bound.
        _, count = self.document.search_triples("", str(RDF.type), root_type)
        return count

    def predicate_objects(
        self, subject: Any
    ) -> Iterable[tuple[GraphTerm, GraphTerm]]:
        subject_term = self.term(subject)
        if subject_term.kind == "literal":
            return
        triples, _ = self.document.search_triples(subject_term.value, "", "")
        for _, pred, obj in triples:
            yield GraphTerm("iri", pred), self._term_from_hdt(obj)

    def objects(self, subject: Any, predicate_iri: str) -> Iterable[GraphTerm]:
        subject_term = self.term(subject)
        if subject_term.kind == "literal":
            return
        triples, _ = self.document.search_triples(
            subject_term.value, predicate_iri, ""
        )
        for _, _, obj in triples:
            yield self._term_from_hdt(obj)

    def term(self, node: Any) -> GraphTerm:
        if isinstance(node, GraphTerm):
            return node
        if isinstance(node, Literal):
            return GraphTerm("literal", str(node))
        if isinstance(node, BNode):
            return GraphTerm("bnode", f"_:{node}")
        if isinstance(node, URIRef):
            return GraphTerm("iri", str(node))
        return self._term_from_hdt(str(node))

    def _term_from_hdt(self, text: str) -> GraphTerm:
        if text.startswith("_:"):
            return GraphTerm("bnode", text)
        if text.startswith('"'):
            try:
                node = from_n3(text)
            except Exception:
                return GraphTerm("literal", text)
            if isinstance(node, Literal):
                return GraphTerm("literal", str(node))
            return GraphTerm("literal", text)
        return GraphTerm("iri", text)


def graph_reader(graph: Graph) -> GraphReader:
    if hasattr(graph.store, "hdt_document"):
        return HDTGraphReader(graph)
    return RDFLibGraphReader(graph)
