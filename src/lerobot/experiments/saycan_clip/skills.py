from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SkillCandidate:
    """Normalized representation of a high-level skill candidate.

    Phase A: produced from CLIP-scored text candidates.
    Phase B: will be mapped to executable robot skills.
    """

    name: str
    score: float
    raw_text: str
    params: dict[str, str] | None = None


SkillFn = Callable[[SkillCandidate], None]


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillFn] = {}

    def register(self, name: str, fn: SkillFn) -> None:
        self._skills[name] = fn

    def get(self, name: str) -> SkillFn | None:
        return self._skills.get(name)

    def known(self) -> list[str]:
        return sorted(self._skills)

