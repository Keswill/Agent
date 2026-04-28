

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Action:

    kind: str
    target_id: str
    destination: Optional[str] = None




_PLAN_RE = re.compile(
    r"^\s*(?P<kind>pick|place)\s*\(\s*(?P<target>[A-Za-z0-9_\-]+)\s*(?:,\s*(?P<dest>[A-Za-z0-9_\-]+)\s*)?\)\s*$"
)


def parse_plan_strings(
    plan_strings: List[str],
    default_place_bin: str = "LEFT_BIN",
) -> List[Action]:
    actions: List[Action] = []
    for i, raw in enumerate(plan_strings):
        m = _PLAN_RE.match(raw)
        if not m:
            raise ValueError(
                "plan entry #{} {!r} does not match 'pick(id)' or 'place(id[, bin])'".format(i, raw)
            )
        kind = m.group("kind").lower()
        target = m.group("target")
        dest = m.group("dest")
        if kind == "place":
            actions.append(Action(kind="place", target_id=target,
                                  destination=dest or default_place_bin))
        else:
            actions.append(Action(kind="pick", target_id=target))
    return actions
