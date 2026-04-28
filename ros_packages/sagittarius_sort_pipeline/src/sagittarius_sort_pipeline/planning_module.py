from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set


_CATEGORY_ALIASES = {
    "vegetable": {"vegetable", "vegetables", "veggie", "veggies"},
    "fruit": {"fruit", "fruits"},
}

_COLORS = {
    "red",
    "blue",
    "green",
    "yellow",
    "white",
    "black",
    "orange",
    "purple",
    "pink",
    "gray",
    "grey",
}

_STOPWORDS = {
    "a",
    "an",
    "all",
    "and",
    "any",
    "bin",
    "into",
    "move",
    "object",
    "objects",
    "out",
    "pick",
    "place",
    "put",
    "sort",
    "the",
    "to",
}


class PlanningModule:
    def __init__(self, default_policy: str = "all_if_no_filter"):
        if default_policy not in {"all_if_no_filter", "none_if_no_filter"}:
            raise ValueError("Unsupported default_policy={!r}".format(default_policy))
        self.default_policy = default_policy

    def plan(self, instruction: str, objects: Dict[str, Dict[str, Any]]) -> List[str]:
        filters = _parse_filters(instruction)
        selected = [obj_id for obj_id, attrs in objects.items() if _matches(attrs, filters)]

        if not filters.has_any_filter and self.default_policy == "all_if_no_filter":
            selected = list(objects.keys())

        plan: List[str] = []
        for obj_id in selected:
            plan.append("pick({})".format(obj_id))
            plan.append("place({})".format(obj_id))
        return plan


class _Filters:
    def __init__(
        self,
        *,
        categories: Set[str],
        colors: Set[str],
        keywords: Set[str],
        need_fridge: Optional[bool],
    ):
        self.categories = categories
        self.colors = colors
        self.keywords = keywords
        self.need_fridge = need_fridge

    @property
    def has_any_filter(self) -> bool:
        return bool(self.categories or self.colors or self.keywords or self.need_fridge is not None)


def _parse_filters(instruction: str) -> _Filters:
    text = instruction.lower()
    categories = {
        category
        for category, aliases in _CATEGORY_ALIASES.items()
        if any(re.search(r"\b{}\b".format(re.escape(alias)), text) for alias in aliases)
    }
    colors = {color for color in _COLORS if re.search(r"\b{}\b".format(re.escape(color)), text)}
    need_fridge: Optional[bool] = None
    if any(phrase in text for phrase in ("need fridge", "needs fridge", "refrigerated", "fridge")):
        need_fridge = True
    if any(phrase in text for phrase in ("no fridge", "not refrigerated", "do not refrigerate")):
        need_fridge = False

    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text))
    alias_words = set().union(*_CATEGORY_ALIASES.values())
    keywords = words - _STOPWORDS - alias_words - _COLORS
    return _Filters(
        categories=categories,
        colors=colors,
        keywords=keywords,
        need_fridge=need_fridge,
    )


def _matches(attrs: Dict[str, Any], filters: _Filters) -> bool:
    if filters.categories and _norm(attrs.get("category")) not in filters.categories:
        return False
    if filters.colors and _norm(attrs.get("color")) not in filters.colors:
        return False
    if filters.need_fridge is not None and bool(attrs.get("need_fridge", False)) != filters.need_fridge:
        return False
    if filters.keywords and not _contains_keyword(attrs, filters.keywords):
        return False
    return filters.has_any_filter


def _contains_keyword(attrs: Dict[str, Any], keywords: Iterable[str]) -> bool:
    haystack = " ".join(str(attrs.get(key, "")) for key in ("name", "category", "shape", "color")).lower()
    return any(keyword in haystack for keyword in keywords)


def _norm(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None
