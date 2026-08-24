from urllib.parse import quote

import httpx
from flask import Blueprint, jsonify, render_template, request
from pydantic import ValidationError

from ..core.errors import URINotFoundError, unwrap_qdrant_error
from ..core.models import Query, build_query
from ..core.query import run_similarity_search
from ..core.results import ResultRow
from ._flask import get_ctx

api = Blueprint("api", __name__)
web = Blueprint("web", __name__)


def serialize_row(row: ResultRow) -> dict:
    # `payload` mirrors the fields templates and API consumers read off
    # the old Qdrant payload, so the response shape is backend-agnostic.
    return {
        "id": row.id,
        "score": row.score,
        "payload": {
            "label": row.label,
            "graph": row.graph,
            "repr": row.repr,
            "iri": row.iris,
            "iri_count": row.iri_count,
        },
        "iris": row.iris,
        "iri_count": row.iri_count,
        "primary_uri": row.primary_uri,
        "encoded_uri": quote(row.primary_uri, safe=""),
    }


def parse_error(e: Exception):
    inner = unwrap_qdrant_error(e)
    msg = str(inner)
    match inner:
        case URINotFoundError():
            status = 404
        case httpx.ConnectError():
            status = 500
            msg = "Could not connect to Qdrant server"
        case ValueError():
            status = 400
        case _:
            status = 500
    return msg, status


@api.post("/query")
def post_query():
    data = request.get_json(silent=True) or {}

    try:
        q = Query.model_validate(data)
    except ValidationError as e:
        return jsonify({"error": "invalid request", "details": e.errors()}), 400

    ctx = get_ctx()

    try:
        result = run_similarity_search(
            ctx,
            query_obj=q,
        )
    except Exception as e:
        msg, status = parse_error(e)
        return jsonify({"error": msg}), status

    return jsonify({"results": [serialize_row(row) for row in result.rows]})


@web.get("/")
def index():
    ctx = get_ctx()

    feature_type = request.args.get("type", "Text")
    feature_value = request.args.get("value", "")
    graphs = ctx.graphs

    graph_mode = request.args.get("graph-mode", "include")
    selected_graphs = request.args.getlist("graph")

    return render_template(
        "index.html",
        feature_type=feature_type,
        feature_value=feature_value,
        graphs=graphs,
        graph_mode=graph_mode,
        selected_graphs=selected_graphs,
        backend_name=ctx.settings.search_backend,
    )


@web.post("/query-view")
def post_query_view():
    form = request.form

    try:
        q = build_query(
            feature_type=form.get("feat_type", ""),
            value=form.get("feat_value", ""),
            include_graphs=form.getlist("include_graphs"),
            exclude_graphs=form.getlist("exclude_graphs"),
            limit=form.get("limit", 10),
            offset=form.get("offset", 0),
        )
    except ValidationError:
        return render_template(
            "partials/results_table.html",
            results=[],
            error="Invalid query.",
        ), 400

    ctx = get_ctx()
    try:
        result = run_similarity_search(
            ctx,
            query_obj=q,
        )
    except Exception as e:
        msg, status = parse_error(e)
        return render_template(
            "partials/results_table.html",
            results=[],
            error=msg,
        ), status

    results = [serialize_row(row) for row in result.rows]
    return render_template(
        "partials/results_table.html",
        results=results,
        query=q,
    )
