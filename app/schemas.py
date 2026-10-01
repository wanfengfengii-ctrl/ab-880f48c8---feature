"""Request/response schemas with strict input validation.

Illegal payloads (wrong counts, non-positive values, duplicate ids, ...) are
rejected with HTTP 422 *before* the solver is invoked.
"""

from typing import List, Optional

from pydantic import (BaseModel, ConfigDict, Field, model_serializer,
                      model_validator)


class StrictModel(BaseModel):
    # Strict typing: int fields reject strings/floats/booleans rather than
    # coercing them, so illegal payloads never reach the solver.
    model_config = ConfigDict(extra="forbid", strict=True)


class ArmIn(StrictModel):
    id: str = Field(..., min_length=1)
    x: int
    y: int
    max_extension: int = Field(..., ge=1)


class TargetIn(StrictModel):
    id: str = Field(..., min_length=1)
    x: int
    y: int
    priority: int = Field(..., ge=1)


class DualFibreGroupIn(StrictModel):
    """One dual-fibre observation group: two *different* existing targets
    that must be allocated together (two arms, simultaneously) or not at all.
    A target may belong to at most one group in the request."""
    target_a: str = Field(..., min_length=1)
    target_b: str = Field(..., min_length=1)


class AdjudicateRequest(StrictModel):
    arms: List[ArmIn] = Field(..., min_length=6, max_length=12)
    targets: List[TargetIn] = Field(..., min_length=6, max_length=16)
    clearance: int = Field(..., ge=1)
    minimum_allocations: int = Field(..., ge=1)
    # Optional: zero (omitted) to six disjoint dual-fibre observation groups.
    # When omitted the request/response stay byte-for-byte compatible with
    # the ungrouped API.
    dual_fibre_groups: Optional[List[DualFibreGroupIn]] = Field(
        None, min_length=1, max_length=6)

    @model_validator(mode="after")
    def _unique_ids_and_groups(self) -> "AdjudicateRequest":
        arm_ids = [a.id for a in self.arms]
        target_ids = [t.id for t in self.targets]
        if len(set(arm_ids)) != len(arm_ids):
            dup = sorted({x for x in arm_ids if arm_ids.count(x) > 1})
            raise ValueError(f"duplicate arm ids: {dup}")
        if len(set(target_ids)) != len(target_ids):
            dup = sorted({x for x in target_ids if target_ids.count(x) > 1})
            raise ValueError(f"duplicate target ids: {dup}")
        groups = self.dual_fibre_groups
        if groups:
            target_set = set(target_ids)
            seen: set = set()
            for idx, g in enumerate(groups):
                if g.target_a == g.target_b:
                    raise ValueError(
                        f"dual_fibre_groups[{idx}]: target_a and target_b "
                        f"must be different targets")
                for name, tid in (("target_a", g.target_a),
                                  ("target_b", g.target_b)):
                    if tid not in target_set:
                        raise ValueError(
                            f"dual_fibre_groups[{idx}].{name}={tid!r} does "
                            f"not reference a submitted target")
                    if tid in seen:
                        raise ValueError(
                            f"dual_fibre_groups[{idx}]: target {tid!r} is "
                            f"reused across groups (targets must belong to at "
                            f"most one group)")
                    seen.add(tid)
        return self


class AssignmentOut(BaseModel):
    arm_id: str
    target_id: str
    extension: float
    extension_sq: int


class GroupAllocationOut(BaseModel):
    arm_id: str
    target_id: str


class DualFibreGroupOut(BaseModel):
    """Atomic per-group result: either ``allocated=true`` with exactly two
    allocations (and their arms), or ``allocated=false`` with none."""
    group_index: int
    targets: List[str]
    allocated: bool
    allocations: List[GroupAllocationOut]
    reason: Optional[str] = None


class PairClearance(BaseModel):
    arm_a: str
    target_a: str
    arm_b: str
    target_b: str
    distance: float
    distance_exact: str
    distance_squared_exact: str
    required_clearance: int
    satisfied: bool
    witness: dict


class WitnessPair(BaseModel):
    arm_a: str
    target_a: str
    arm_b: str
    target_b: str
    distance: float
    distance_exact: str
    required_clearance: int
    witness: dict


class DualFibreShortfall(BaseModel):
    groups_submitted: int
    groups_allocated: int
    groups_unallocated: int
    grouped_max_allocations: int
    ungrouped_max_allocations: int
    allocations_lost_to_atomicity: int
    minimum_blocked_by_atomicity: bool


class ShortfallWitness(BaseModel):
    blocking_pair: Optional[WitnessPair] = None
    capacity_limit: int
    reachable_arm_target_links: int
    ungrouped_max_allocations: Optional[int] = None
    dual_fibre_shortfall: Optional[DualFibreShortfall] = None

    @model_serializer(mode="wrap")
    def _serialize_compat(self, handler):
        data = handler(self)
        for key in ("ungrouped_max_allocations", "dual_fibre_shortfall"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class AdjudicateResponse(BaseModel):
    status: str  # "ok" | "minimum_not_met"
    feasible: bool
    reason: Optional[str] = None
    objective: dict
    assignments: List[AssignmentOut]
    unassigned_arms: List[str]
    unassigned_targets: List[str]
    arm_lengths: dict
    clearance: dict
    limits: dict
    witness: Optional[ShortfallWitness] = None
    # Per-group atomic status; omitted entirely when no groups are submitted.
    dual_fibre_groups: Optional[List[DualFibreGroupOut]] = None

    @model_serializer(mode="wrap")
    def _serialize_compat(self, handler):
        data = handler(self)
        if not data.get("dual_fibre_groups"):
            data.pop("dual_fibre_groups", None)
        return data

