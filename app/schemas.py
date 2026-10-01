"""Request/response schemas with strict input validation.

Illegal payloads (wrong counts, non-positive values, duplicate ids, ...) are
rejected with HTTP 422 *before* the solver is invoked.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class FibreGroupIn(StrictModel):
    id: str = Field(..., min_length=1)
    target_a: str = Field(..., min_length=1)
    target_b: str = Field(..., min_length=1)


class AdjudicateRequest(StrictModel):
    arms: List[ArmIn] = Field(..., min_length=6, max_length=12)
    targets: List[TargetIn] = Field(..., min_length=6, max_length=16)
    clearance: int = Field(..., ge=1)
    minimum_allocations: int = Field(..., ge=1)
    # Optional dual-fibre observation groups (1..6).  Each group references
    # two distinct targets; a target may belong to at most one group.
    fibre_groups: Optional[List[FibreGroupIn]] = Field(
        default=None, min_length=1, max_length=6)

    @model_validator(mode="after")
    def _unique_ids(self) -> "AdjudicateRequest":
        arm_ids = [a.id for a in self.arms]
        target_ids = [t.id for t in self.targets]
        if len(set(arm_ids)) != len(arm_ids):
            dup = sorted({x for x in arm_ids if arm_ids.count(x) > 1})
            raise ValueError(f"duplicate arm ids: {dup}")
        if len(set(target_ids)) != len(target_ids):
            dup = sorted({x for x in target_ids if target_ids.count(x) > 1})
            raise ValueError(f"duplicate target ids: {dup}")
        if self.fibre_groups is not None:
            group_ids = [g.id for g in self.fibre_groups]
            if len(set(group_ids)) != len(group_ids):
                dup = sorted({x for x in group_ids
                              if group_ids.count(x) > 1})
                raise ValueError(f"duplicate fibre group ids: {dup}")
            target_set = set(target_ids)
            seen: List[str] = []
            for g in self.fibre_groups:
                if g.target_a == g.target_b:
                    raise ValueError(
                        f"fibre group {g.id!r} must reference two distinct "
                        "targets")
                for t in (g.target_a, g.target_b):
                    if t not in target_set:
                        raise ValueError(
                            f"fibre group {g.id!r} references unknown target "
                            f"{t!r}")
                    if t in seen:
                        raise ValueError(
                            f"target {t!r} is reused by more than one fibre "
                            "group; targets must not be shared across groups")
                    seen.append(t)
        return self


class AssignmentOut(BaseModel):
    arm_id: str
    target_id: str
    extension: float
    extension_sq: int


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


class ShortfallWitness(BaseModel):
    blocking_pair: Optional[WitnessPair] = None
    capacity_limit: int
    reachable_arm_target_links: int
    dual_fibre: Optional["DualFibreWitness"] = None


class FibreGroupMemberOut(BaseModel):
    group_id: str
    slot: str  # "a" | "b"
    target_id: str
    allocated: bool
    arm_id: Optional[str] = None
    extension: Optional[float] = None
    extension_sq: Optional[int] = None


class FibreGroupOut(BaseModel):
    group_id: str
    status: str  # "allocated" | "unallocated"
    members: List[FibreGroupMemberOut]


class DualFibreSplitGroup(BaseModel):
    group_id: str
    allocated_target: str
    allocated_arm: str
    unallocated_target: str


class DualFibreWitness(BaseModel):
    constraint: str
    groups_submitted: int
    maximum_without_atomicity: int
    maximum_with_atomicity: int
    allocations_blocked_by_atomicity: int
    would_be_split_groups: List[DualFibreSplitGroup]
    note: str


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
    fibre_groups: Optional[List[FibreGroupOut]] = None
