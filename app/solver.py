"""Fibre-arm / target allocation solver.

Given fibre arms (integer base + maximum extension) and unique targets with
positive integer priorities, find the lexicographically optimal assignment:

    1. maximise the number of allocations;
    2. maximise the sum of priorities of allocated targets;
    3. minimise the sum of squared arm extensions;
    4. the stable sequence: among remaining ties, prefer allocating the
       earlier arm (submission order), and to the lowest-indexed target.
       Concretely minimise the arm-order tuple whose entry is the assigned
       target index and ``n_targets`` for an unused arm.

Hard constraints:

* an arm connects to at most one reachable target (distance <= reach);
* a target is used at most once;
* the closed segment of every pair of allocations keeps at least ``clearance``
  distance from every other allocation segment;
* optional *dual-fibre groups*: ``groups`` holds disjoint pairs of target
  indices.  The two targets of a group are indivisible: either both are
  allocated (to two distinct arms, simultaneously) or neither is.  Targets
  outside groups are still allocated individually.  Group atomicity is purely
  a feasibility constraint - the four-level objective order is unchanged.

Algorithm
---------
The whole lexicographic objective is encoded as one *integer* edge weight
(with dominance constants derived from the instance bounds), so a single
maximum-weight bipartite matching (Hungarian method) solves the relaxed
problem where pairwise collisions and half-groups are ignored.
Branch-and-bound then branches on every remaining violation:

* collision: if the relaxed matching contains a conflicting pair of
  allocations, every feasible solution omits at least one of the two edges;
* half-group: a group has exactly one target matched; either that edge is
  dropped, or the partner target is forced onto one of its candidate arms.

The relaxed matching value is an upper bound used for pruning.  All geometry
stays exact (Fraction).

A scaled weight (factor ``group_scale``) reserves a below-stability digit that
rewards complete groups in the relaxation only: it guides the matching away
from half-groups, while two atom-valid solutions are still ordered exactly by
the original objective (the bonus ``K`` is bounded by ``group_scale - 1`` and
any real objective difference is a multiple of ``group_scale``).
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .geometry import segment_distance_sq

TargetId = str
ArmId = str
Edge = Tuple[int, int]  # (arm index, target index)
Group = Tuple[int, int]  # two distinct, globally disjoint target indices


@dataclass(frozen=True)
class Target:
    id: TargetId
    x: int
    y: int
    priority: int


@dataclass(frozen=True)
class Arm:
    id: ArmId
    x: int
    y: int
    max_extension: int


@dataclass
class SolveResult:
    assignments: List[dict]
    unassigned_arms: List[ArmId]
    unassigned_targets: List[TargetId]
    arm_lengths: Dict[ArmId, float]
    pair_clearances: List[dict]
    minimum_clearance: Optional[float]
    num_allocations: int
    priority_sum: int
    extension_sq_sum: int
    reachable_count: int
    minimum_requested: int
    feasible: bool
    groups: List[dict]
    ungrouped_maximum: Optional[int] = None
    reason: Optional[str] = None
    witness: Optional[dict] = None


INF = 10**100


class Solver:
    def __init__(
        self,
        arms: Sequence[Arm],
        targets: Sequence[Target],
        clearance: int,
        minimum_allocations: int,
        groups: Optional[Sequence[Group]] = None,
    ) -> None:
        self.arms = list(arms)
        self.targets = list(targets)
        self.clearance = clearance
        self.minimum_allocations = minimum_allocations
        self.n_arms = len(self.arms)
        self.n_targets = len(self.targets)
        self.clearance_sq = Fraction(clearance * clearance)

        # Dual-fibre groups (validated disjoint pairs of target indices).
        self.groups: List[Group] = [(a, b) for a, b in (groups or [])]
        self.group_of: List[Optional[int]] = [None] * self.n_targets
        for gi, (a, b) in enumerate(self.groups):
            self.group_of[a] = gi
            self.group_of[b] = gi

        # len2[i][j] is None when target j is unreachable for arm i.
        self.len2: List[List[Optional[int]]] = [
            [None] * self.n_targets for _ in range(self.n_arms)
        ]
        self.edges: List[Edge] = []
        for i, arm in enumerate(self.arms):
            for j, tgt in enumerate(self.targets):
                d2 = (arm.x - tgt.x) ** 2 + (arm.y - tgt.y) ** 2
                if d2 <= arm.max_extension**2:
                    self.len2[i][j] = d2
                    self.edges.append((i, j))
        self.total_reachable = len(self.edges)

        # Candidate arms per grouped target (needed for half-group branching).
        self.cand_arms: List[List[int]] = [
            [i for i in range(self.n_arms) if self.len2[i][j] is not None]
            for j in range(self.n_targets)
        ]
        # True when a group cannot even be completed without arm sharing
        # (fewer than two reachable arms over its two targets).
        self.group_dead: List[bool] = [
            len(set(self.cand_arms[a]) | set(self.cand_arms[b])) < 2
            for a, b in self.groups
        ]

        self._pair_cache: Dict[Tuple[int, int, int, int],
                               Tuple[bool, Fraction, dict]] = {}
        # Incident edges and lazily built per-edge conflict sets make
        # "anchor edge" work a few set unions instead of repeated Python
        # geometry loops inside branch-and-bound.
        self.edges_by_arm: List[List[Edge]] = [[] for _ in range(self.n_arms)]
        self.edges_by_target: List[List[Edge]] = [[] for _ in range(self.n_targets)]
        for e in self.edges:
            self.edges_by_arm[e[0]].append(e)
            self.edges_by_target[e[1]].append(e)
        self._conflict_cache: Dict[Edge, FrozenSet[Edge]] = {}

        self._make_weights()

    def conflicts_of(self, edge: Edge) -> FrozenSet[Edge]:
        """Edges on other arms whose closed segment is within clearance of
        ``edge``; computed once per edge and cached."""
        cached = self._conflict_cache.get(edge)
        if cached is not None:
            return cached
        ei, ej = edge
        hit: set = set()
        for (k, h) in self.edges:
            if k == ei:
                continue
            a, b = (ei, k) if ei < k else (k, ei)
            t1, t2 = (ej, h) if ei < k else (h, ej)
            if self.pair_info(a, t1, b, t2)[0]:
                hit.add((k, h))
        result = frozenset(hit)
        self._conflict_cache[edge] = result
        return result

    # ------------------------------------------------------------------
    # Lexicographic objective as one integer edge weight
    # ------------------------------------------------------------------
    def _make_weights(self) -> None:
        n, m = self.n_arms, self.n_targets
        base = m + 1  # stability digit base (digits 0..m)

        # Stability magnitude bound: sum of digit values * place value.
        place = [base ** (n - 1 - i) for i in range(n)]
        s_max = m * sum(place)

        # Extension magnitude bound (worst case all arms at max reachable d2).
        e_max = sum(
            max((d2 for d2 in row if d2 is not None), default=0)
            for row in self.len2
        )
        p_max = n * max((t.priority for t in self.targets), default=1)

        # Maximise: C3*count + C2*priority - C1*ext2 + stability_term.
        self.c1 = s_max + 1
        self.c2 = e_max * self.c1 + s_max + 1
        self.c3 = p_max * self.c2 + e_max * self.c1 + s_max + 1
        self._place = place

        # Scale the objective by (2*n_groups + 1) to reserve a digit below
        # the stability term.  The digit holds the count of matched grouped
        # targets (2 per completed group, range [0, 2*n_groups]); it only
        # guides the relaxation: since every raw objective difference is a
        # multiple of the scale and the digit is strictly smaller than the
        # scale, atom-valid solutions are ordered by the raw objective alone
        # (the digit breaks raw ties toward more complete groups, never
        # hurting an atom-valid optimum).
        self.group_scale = 2 * len(self.groups) + 1

        self.weight: Dict[Edge, int] = {}
        for i, j in self.edges:
            stability_gain = (m - j) * place[i]  # vs. leaving arm i unused
            grouped = 1 if self.group_of[j] is not None else 0
            self.weight[(i, j)] = self.group_scale * (
                self.c3
                + self.c2 * self.targets[j].priority
                - self.c1 * self.len2[i][j]
                + stability_gain
            ) + grouped

    def matching_weight(self, assignment: Sequence[int]) -> int:
        return sum(self.weight[(i, j)] for i, j in enumerate(assignment) if j >= 0)

    def complete_groups(self, assignment: Sequence[int]) -> int:
        used = {j for j in assignment if j >= 0}
        return sum(1 for a, b in self.groups if a in used and b in used)

    # ------------------------------------------------------------------
    # Pairwise segment feasibility (exact arithmetic)
    # ------------------------------------------------------------------
    def pair_info(self, i: int, j: int, k: int, h: int):
        """Info for simultaneous allocations (i->j),(k->h), with i < k."""
        key = (i, j, k, h)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached

        arm_i, tgt_j = self.arms[i], self.targets[j]
        arm_k, tgt_h = self.arms[k], self.targets[h]
        dist_sq, u, v = segment_distance_sq(
            (arm_i.x, arm_i.y), (tgt_j.x, tgt_j.y),
            (arm_k.x, arm_k.y), (tgt_h.x, tgt_h.y),
        )
        witness = {
            "point_on_first": [
                _frac_str(arm_i.x + u * (tgt_j.x - arm_i.x)),
                _frac_str(arm_i.y + u * (tgt_j.y - arm_i.y)),
            ],
            "point_on_second": [
                _frac_str(arm_k.x + v * (tgt_h.x - arm_k.x)),
                _frac_str(arm_k.y + v * (tgt_h.y - arm_k.y)),
            ],
        }
        result = (dist_sq < self.clearance_sq, dist_sq, witness)
        self._pair_cache[key] = result
        return result

    def _first_conflict(self, used_edges: List[Edge]
                        ) -> Optional[Tuple[Edge, Edge]]:
        used_edges = sorted(used_edges)
        used_set = set(used_edges)
        for e in used_edges:
            hit = self.conflicts_of(e) & used_set
            if hit:
                f = min(hit)
                return (e, f) if e < f else (f, e)
        return None

    def _pick_half_group(self, used_edges: List[Edge], forbidden: FrozenSet[Edge]
                         ) -> Optional[Tuple[int, Edge]]:
        """Half-group with the fewest ways to place the partner target
        without colliding with the lone matched edge (narrowest fallback)."""
        matched_targets = {j: (i, j) for i, j in used_edges}
        best: Optional[Tuple[int, int, Edge]] = None
        for gi, (a, b) in enumerate(self.groups):
            in_a = a in matched_targets
            in_b = b in matched_targets
            if in_a == in_b:
                continue
            j = a if in_a else b
            ei, ej = matched_targets[j]
            partner = b if j == a else a
            options = 0
            blocked = self.conflicts_of(matched_targets[j])
            for k in self.cand_arms[partner]:
                if k == ei or (k, partner) in forbidden:
                    continue
                if (k, partner) not in blocked:
                    options += 1
            cand = (options, gi, matched_targets[j])
            if best is None or cand[:2] < best[:2]:
                best = cand
        return None if best is None else (best[1], best[2])

    def _branch_edge(self, chosen: List[Edge]) -> Edge:
        """Pick the chosen edge participating in the most internal clashes;
        ties broken by arm then target order."""
        chosen_set = set(chosen)
        degrees = {}
        for e in chosen:
            for f in self.conflicts_of(e) & chosen_set:
                degrees[e] = degrees.get(e, 0) + 1
        if not degrees:
            return chosen[0]
        return min(degrees, key=lambda e: (-degrees[e], e))

    # ------------------------------------------------------------------
    # Relaxed maximum-weight matching with a forbidden-edge set
    # ------------------------------------------------------------------
    def _relaxed_match(self, forbidden: FrozenSet[Edge]
                       ) -> Tuple[int, List[Edge]]:
        n, m = self.n_arms, self.n_targets
        size = max(n, m, 1)
        cost = [[0] * size for _ in range(size)]
        for (i, j), w in self.weight.items():
            if (i, j) not in forbidden:
                cost[i][j] = -w  # Hungarian minimises
        row_owner, _ = _hungarian(cost)
        chosen: List[Edge] = []
        total = 0
        for i in range(n):
            j = row_owner[i]
            if 0 <= j < m and self.len2[i][j] is not None and (i, j) not in forbidden:
                chosen.append((i, j))
                total += self.weight[(i, j)]
        return total, chosen

    # ------------------------------------------------------------------
    # Branch and bound on collision pairs and half-groups
    # ------------------------------------------------------------------
    def solve(self) -> Tuple[Tuple[int, ...], bool, Optional[dict]]:
        # Feasible incumbent from deterministic greedy seeds (enables pruning
        # at the root); then exact branching proves optimality.  Every seed
        # and every accepted leaf is collision-free *and* group-atomic.
        incumbent = self._greedy_seed()
        best_value = self.matching_weight(incumbent)
        best_assignment = incumbent[:]
        stats = {"nodes": 0}
        memo: Dict[FrozenSet[Edge], Tuple[int, List[Edge]]] = {}

        # Groups that cannot be completed even ignoring geometry (fewer than
        # two reachable arms over their two targets) can never contribute:
        # forbid every edge incident to their targets up front.
        root_forbidden: set = set()
        for gi, (a, b) in enumerate(self.groups):
            if self.group_dead[gi]:
                for jj in (a, b):
                    for i in range(self.n_arms):
                        if self.len2[i][jj] is not None:
                            root_forbidden.add((i, jj))

        def include_anchor(forbidden: FrozenSet[Edge], edge: Edge
                          ) -> Optional[FrozenSet[Edge]]:
            """Restrict to solutions that use ``edge``: forbid every
            alternative on its arm/target and every edge in the whole graph
            that collides with it.  ``None`` if contradictory."""
            if edge in forbidden:
                return None
            ei, ej = edge
            forced = (set(forbidden)
                      | set(self.edges_by_arm[ei])
                      | set(self.edges_by_target[ej])
                      | set(self.conflicts_of(edge)))
            forced.discard(edge)
            return frozenset(forced)

        def recurse(forbidden: FrozenSet[Edge]) -> None:
            nonlocal best_value, best_assignment
            stats["nodes"] += 1
            cached = memo.get(forbidden)
            if cached is not None:
                bound, chosen = cached
            else:
                bound, chosen = self._relaxed_match(forbidden)
                memo[forbidden] = (bound, chosen)
            if bound <= best_value:
                return  # cannot improve the incumbent

            # Half-groups are branched before collisions: anchoring the lone
            # allocation removes its arm, its target and all colliding edges
            # at once, which narrows the remaining collision search far more
            # than branching a collision first.
            half = self._pick_half_group(chosen, forbidden)
            if half is not None:
                _, edge = half
                ei, ej = edge
                partner = self._partner_of(ej)
                anchored = include_anchor(forbidden, edge)
                if anchored != forbidden:
                    # Keeping the edge: descend; the relaxation at the
                    # anchored node either completes the group or keeps
                    # presenting half-groups (forbidden strictly grows, so
                    # this always terminates).
                    recurse(anchored)
                    if best_value == bound:
                        return
                else:
                    # The edge is already effectively anchored (no other
                    # reachable edge on its arm/target, no clash to forbid)
                    # yet the partner stays unmatched: enumerate the arm that
                    # carries the partner.
                    for k in self.cand_arms[partner]:
                        if k == ei:
                            continue
                        forced = include_anchor(forbidden, (k, partner))
                        if (forced is not None and forced != forbidden
                                and edge not in forced):
                            recurse(forced)
                            if best_value == bound:
                                return
                # Dropping the lone edge: the group is absent in this subtree.
                recurse(forbidden | {edge})
                return

            # Collision violations: binary include/exclude on the edge that
            # participates in the most clashes.
            if self._first_conflict(chosen) is not None:
                edge = self._branch_edge(chosen)
                forced = include_anchor(forbidden, edge)
                if forced is not None:
                    recurse(forced)
                    if best_value == bound:
                        return
                recurse(forbidden | {edge})
                return

            # Relaxed optimum is collision-free and group-atomic -> true
            # optimum of this node.
            assignment = [-1] * self.n_arms
            for i, j in chosen:
                assignment[i] = j
            best_value = bound
            best_assignment = assignment

        recurse(frozenset(root_forbidden))
        self.solve_stats = stats

        best_tuple = tuple(best_assignment)
        count = sum(1 for x in best_tuple if x >= 0)
        witness = None
        if count < self.minimum_allocations:
            witness = self._shortfall_witness(best_tuple)
        return best_tuple, count >= self.minimum_allocations, witness

    def _partner_of(self, target: int) -> int:
        """The other target index in the group containing ``target``."""
        gi = self.group_of[target]
        a, b = self.groups[gi]
        return b if target == a else a

    # ------------------------------------------------------------------
    # Greedy feasible seeds (always collision-free and group-atomic)
    # ------------------------------------------------------------------
    def _edge_ok(self, i: int, j: int, placed: List[Edge]) -> bool:
        for k, h in placed:
            a, b = (i, k) if i < k else (k, i)
            t1, t2 = (j, h) if i < k else (h, j)
            if self.pair_info(a, t1, b, t2)[0]:
                return False
        return True

    def _greedy_seed(self) -> List[int]:
        """Best of several deterministic greedy, always-feasible passes.

        Every returned assignment respects group atomicity: grouped targets
        only ever appear as a complete pair placed in one move; targets of a
        group that could not be placed are left unused.
        """
        n, m = self.n_arms, self.n_targets
        candidates: List[List[int]] = []

        def best_placement(gi, assignment, placed):
            """Highest-gain feasible placement of group ``gi`` given the
            current partial assignment: ``((i, j_a_or_b), (k, other))``."""
            a, b = self.groups[gi]
            best = None
            for ja, jb in ((a, b), (b, a)):
                for i in self.cand_arms[ja]:
                    if assignment[i] >= 0 or not self._edge_ok(i, ja, placed):
                        continue
                    for k in self.cand_arms[jb]:
                        if k == i or assignment[k] >= 0:
                            continue
                        if not self._edge_ok(k, jb, placed + [(i, ja)]):
                            continue
                        gain = self.weight[(i, ja)] + self.weight[(k, jb)]
                        # canonical orientation tie-break: a on smaller arm
                        orient = 0 if ja == a else 1
                        key = (-gain, min(i, k), max(i, k), orient)
                        if best is None or key < best[0]:
                            best = (key, (i, ja, k, jb))
            return None if best is None else best[1]

        def seed_phased(group_order, arm_order, target_key):
            """Place whole groups first (in ``group_order``), then greedily
            assign ungrouped targets to the remaining arms."""
            assignment = [-1] * n
            used = [False] * m
            placed: List[Edge] = []
            for gi in group_order:
                if self.group_dead[gi]:
                    continue
                a, b = self.groups[gi]
                if used[a] or used[b]:
                    continue
                mv = best_placement(gi, assignment, placed)
                if mv is not None:
                    i, ja, k, jb = mv
                    assignment[i], assignment[k] = ja, jb
                    used[ja] = used[jb] = True
                    placed += [(i, ja), (k, jb)]
            for i in arm_order:
                if assignment[i] >= 0:
                    continue
                for j in sorted((j for j in range(m)
                                 if self.group_of[j] is None
                                 and self.len2[i][j] is not None and not used[j]),
                                key=lambda j, i=i: target_key(i, j)):
                    if self._edge_ok(i, j, placed):
                        assignment[i] = j
                        used[j] = True
                        placed.append((i, j))
                        break
            return assignment

        def seed_best_move(prefer_groups: bool):
            """Repeatedly take the highest-gain feasible move, choosing
            between single (ungrouped) allocations and whole-group moves."""
            assignment = [-1] * n
            used = [False] * m
            placed: List[Edge] = []
            while True:
                moves = []
                for i in range(n):
                    if assignment[i] >= 0:
                        continue
                    for j in range(m):
                        if used[j] or self.group_of[j] is not None:
                            continue
                        if self.len2[i][j] is not None and \
                                self._edge_ok(i, j, placed):
                            moves.append((self.weight[(i, j)], 0,
                                          (i, j, -1, -1)))
                for gi, (a, b) in enumerate(self.groups):
                    if used[a] or used[b] or self.group_dead[gi]:
                        continue
                    mv = best_placement(gi, assignment, placed)
                    if mv is not None:
                        i, ja, k, jb = mv
                        gain = self.weight[(i, ja)] + self.weight[(k, jb)]
                        moves.append((gain, 1 if prefer_groups else -1, mv))
                if not moves:
                    break
                _, _, mv = max(moves, key=lambda t: (t[0], t[1]))
                i, ja, k, jb = mv
                assignment[i] = ja
                used[ja] = True
                placed.append((i, ja))
                if k >= 0:
                    assignment[k] = jb
                    used[jb] = True
                    placed.append((k, jb))
            return assignment

        constrained = sorted(
            range(n),
            key=lambda i: (sum(1 for d2 in self.len2[i] if d2 is not None), i))
        group_idx = list(range(len(self.groups)))
        candidates.append(seed_phased(group_idx, range(n), lambda i, j: j))
        candidates.append(seed_phased(list(reversed(group_idx)), constrained,
                                      lambda i, j: -self.targets[j].priority))
        candidates.append(seed_best_move(prefer_groups=True))
        candidates.append(seed_best_move(prefer_groups=False))
        return max(candidates, key=self.matching_weight)

    # ------------------------------------------------------------------
    # Shortfall explanation
    # ------------------------------------------------------------------
    def _shortfall_witness(self, assignment: Sequence[int]) -> dict:
        """Structural evidence when the minimum cannot be met: the tightest
        violating pair among individually reachable allocations (if any),
        plus an explicit dual-fibre-group attribution."""
        blocking = None
        best_dist: Optional[Fraction] = None
        for i in range(self.n_arms):
            for k in range(i + 1, self.n_arms):
                for j in range(self.n_targets):
                    if self.len2[i][j] is None:
                        continue
                    for h in range(self.n_targets):
                        if h == j or self.len2[k][h] is None:
                            continue
                        conflict, dist_sq, w = self.pair_info(i, j, k, h)
                        if conflict and (best_dist is None or dist_sq > best_dist):
                            best_dist = dist_sq
                            blocking = {
                                "arm_a": self.arms[i].id,
                                "target_a": self.targets[j].id,
                                "arm_b": self.arms[k].id,
                                "target_b": self.targets[h].id,
                                "distance_exact": _sqrt_str(dist_sq),
                                "distance": float(dist_sq ** Fraction(1, 2)),
                                "required_clearance": self.clearance,
                                "witness": w,
                            }
        witness = {
            "blocking_pair": blocking,
            "capacity_limit": min(self.n_arms, self.n_targets),
            "reachable_arm_target_links": self.total_reachable,
        }
        if self.groups:
            grouped_max = sum(1 for x in assignment if x >= 0)
            ref_solver = Solver(self.arms, self.targets, self.clearance,
                                max(self.minimum_allocations, 1))
            ref_assignment, _, _ = ref_solver.solve()
            ungrouped_max = sum(1 for x in ref_assignment if x >= 0)
            witness["ungrouped_max_allocations"] = ungrouped_max
            witness["dual_fibre_shortfall"] = {
                "groups_submitted": len(self.groups),
                "groups_allocated": self.complete_groups(assignment),
                "groups_unallocated": (len(self.groups)
                                       - self.complete_groups(assignment)),
                "grouped_max_allocations": grouped_max,
                "ungrouped_max_allocations": ungrouped_max,
                "allocations_lost_to_atomicity": max(
                    0, ungrouped_max - grouped_max),
                "minimum_blocked_by_atomicity": (
                    ungrouped_max >= self.minimum_allocations
                    and grouped_max < self.minimum_allocations),
            }
        return witness

    # ------------------------------------------------------------------
    # Result assembly with evidence
    # ------------------------------------------------------------------
    def build_result(self, assignment: Sequence[int], feasible: bool,
                     witness: Optional[dict]) -> SolveResult:
        pairs: List[dict] = []
        used_targets = set()
        arm_lengths: Dict[ArmId, float] = {}
        ext_sq_sum = 0
        prio_sum = 0
        placed: List[Edge] = []
        for i, j in enumerate(assignment):
            if j < 0:
                continue
            arm, tgt = self.arms[i], self.targets[j]
            length = (self.len2[i][j] or 0) ** 0.5
            pairs.append({
                "arm_id": arm.id,
                "target_id": tgt.id,
                "extension": float(length),
                "extension_sq": self.len2[i][j],
            })
            arm_lengths[arm.id] = float(length)
            used_targets.add(j)
            ext_sq_sum += self.len2[i][j] or 0
            prio_sum += tgt.priority
            placed.append((i, j))

        pair_clearances: List[dict] = []
        min_d2: Optional[Fraction] = None
        for idx in range(len(placed)):
            for jdx in range(idx + 1, len(placed)):
                i, j = placed[idx]
                k, h = placed[jdx]
                a, b = (i, k) if i < k else (k, i)
                t1, t2 = (j, h) if i < k else (h, j)
                conflict, dist_sq, w = self.pair_info(a, t1, b, t2)
                dist = dist_sq ** Fraction(1, 2)
                record = {
                    "arm_a": self.arms[a].id,
                    "target_a": self.targets[t1].id,
                    "arm_b": self.arms[b].id,
                    "target_b": self.targets[t2].id,
                    "distance": float(dist),
                    "distance_exact": _sqrt_str(dist_sq),
                    "distance_squared_exact": _frac_str(dist_sq),
                    "required_clearance": self.clearance,
                    "satisfied": not conflict,
                    "witness": w,
                }
                pair_clearances.append(record)
                if min_d2 is None or dist_sq < min_d2:
                    min_d2 = dist_sq

        unassigned_arms = [self.arms[i].id
                           for i, j in enumerate(assignment) if j < 0]
        unassigned_targets = [self.targets[j].id
                              for j in range(self.n_targets)
                              if j not in used_targets]

        # Per-group atomic allocation status (complete pairs or the group as
        # a whole being unallocated - a half-group never appears).
        group_status: List[dict] = []
        arm_for_target = {j: i for i, j in placed}
        for gi, (a, b) in enumerate(self.groups):
            if a in arm_for_target and b in arm_for_target:
                ia, ib = arm_for_target[a], arm_for_target[b]
                group_status.append({
                    "group_index": gi,
                    "targets": [self.targets[a].id, self.targets[b].id],
                    "allocated": True,
                    "allocations": [
                        {"arm_id": self.arms[ia].id,
                         "target_id": self.targets[a].id},
                        {"arm_id": self.arms[ib].id,
                         "target_id": self.targets[b].id},
                    ],
                })
            else:
                group_status.append({
                    "group_index": gi,
                    "targets": [self.targets[a].id, self.targets[b].id],
                    "allocated": False,
                    "allocations": [],
                    "reason": ("structurally uncompletable"
                               if self.group_dead[gi]
                               else "not part of the joint optimum"),
                })

        count = len(pairs)
        ungrouped_maximum: Optional[int] = None
        if not feasible:
            if min(self.n_arms, self.n_targets) < self.minimum_allocations:
                cause = "fewer arms/targets than the requested minimum"
            elif self.total_reachable < self.minimum_allocations:
                cause = "not every arm has a reachable target"
            else:
                cause = ("pairwise segment clearance / reachability restricts "
                         "the compatible set below the requested minimum")
            reason = (
                f"minimum_allocations={self.minimum_allocations} not met: the "
                f"maximum non-colliding reachable allocation has {count} "
                f"pair(s); {cause}"
            )
            if witness is not None and "dual_fibre_shortfall" in witness:
                d = witness["dual_fibre_shortfall"]
                ungrouped_maximum = d["ungrouped_max_allocations"]
                reason += (
                    f"; dual-fibre atomicity: {d['groups_allocated']} of "
                    f"{d['groups_submitted']} group(s) fully allocated, "
                    f"{d['groups_unallocated']} group(s) dropped as a whole, "
                    f"joint optimum {d['grouped_max_allocations']} vs "
                    f"{d['ungrouped_max_allocations']} without groups "
                    f"({d['allocations_lost_to_atomicity']} allocation(s) "
                    f"lost to the all-or-nothing group constraint)"
                )
                if d["minimum_blocked_by_atomicity"]:
                    reason += ("; the requested minimum WOULD be reachable "
                               "without the dual-fibre atomic constraint, so "
                               "the shortfall is caused by group atomicity")
        else:
            reason = None

        return SolveResult(
            assignments=pairs,
            unassigned_arms=unassigned_arms,
            unassigned_targets=unassigned_targets,
            arm_lengths=arm_lengths,
            pair_clearances=pair_clearances,
            minimum_clearance=(float(min_d2 ** Fraction(1, 2))
                               if min_d2 is not None else None),
            num_allocations=count,
            priority_sum=prio_sum,
            extension_sq_sum=ext_sq_sum,
            reachable_count=self.total_reachable,
            minimum_requested=self.minimum_allocations,
            feasible=feasible,
            groups=group_status,
            ungrouped_maximum=ungrouped_maximum,
            reason=reason,
            witness=witness,
        )


# ----------------------------------------------------------------------
# Hungarian method (minimisation, square integer cost matrix)
# ----------------------------------------------------------------------
def _hungarian(cost: List[List[int]]) -> Tuple[List[int], List[int]]:
    """Matched column per row and matched row per column (1-indexed
    potentials formulation)."""
    n = len(cost)
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)  # p[col] = row matched to column
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used_col = [False] * (n + 1)
        while True:
            used_col[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used_col[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used_col[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    row_owner = [-1] * n
    col_owner = [-1] * n
    for j in range(1, n + 1):
        i = p[j]
        if i >= 1:
            row_owner[i - 1] = j - 1
            col_owner[j - 1] = i - 1
    return row_owner, col_owner


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------
def _frac_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _sqrt_str(value: Fraction) -> str:
    """Exact rendering of a distance whose square is the rational ``value``."""
    if value == 0:
        return "0"
    # Render rational roots directly when the fraction is a perfect square.
    num_sq = _isqrt(value.numerator)
    den_sq = _isqrt(value.denominator)
    if num_sq * num_sq == value.numerator and den_sq * den_sq == value.denominator:
        q = Fraction(num_sq, den_sq)
        return _frac_str(q)
    return f"sqrt({_frac_str(value)})"


def _isqrt(n: int) -> int:
    """Integer square root (floor) via Newton's method."""
    if n < 2:
        return n
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x
