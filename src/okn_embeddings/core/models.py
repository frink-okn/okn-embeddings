from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator


class TextFeature(BaseModel):
    type: Literal["text"]
    value: str


class NodeFeature(BaseModel):
    type: Literal["node"]
    value: str


Feature = Annotated[
    TextFeature | NodeFeature,
    Field(..., discriminator="type"),
]


class Query(BaseModel):
    feature: Feature
    include_graphs: list[str] | None = None
    exclude_graphs: list[str] | None = None
    limit: int = 10
    offset: int = 0

    @model_validator(mode="after")
    def validate_graph_modes(self):
        if self.include_graphs and self.exclude_graphs:
            raise ValueError(
                "Only one of include_graphs or exclude_graphs may be set"
            )
        return self


def build_feature(feature_type: str, value: str) -> Feature:
    """Build (and validate) a text/node feature from raw parts.

    Shared by the CLI and web routes to turn raw input into the discriminated
    feature union. Raises pydantic ``ValidationError`` on an unknown type.
    """
    adapter = TypeAdapter(Feature)
    return adapter.validate_python({"type": feature_type, "value": value})


def build_query(
    feature_type: str,
    value: str,
    include_graphs: list[str] | None = None,
    exclude_graphs: list[str] | None = None,
    limit: int | str = 10,
    offset: int | str = 0,
) -> Query:
    """Assemble (and validate) a Query from individual parts.

    Shared by the CLI and the HTMX form route so the discriminated feature
    union and the include/exclude mutual-exclusion check are applied
    identically. Raises pydantic ``ValidationError`` on bad input.
    """
    data: dict = {
        "feature": build_feature(feature_type, value),
        "limit": limit,
        "offset": offset,
    }

    if include_graphs:
        data["include_graphs"] = include_graphs

    if exclude_graphs:
        data["exclude_graphs"] = exclude_graphs

    return Query.model_validate(data)
