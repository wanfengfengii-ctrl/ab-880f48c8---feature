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
* optional *dual-fibre groups*: the two targets of a submitted group are
  atomic -- either both are allocated (to two distinct arms, one per arm) or
  neither is.  Targets never belong to more than one group, and ungrouped
  targets can still be allocated individually.

Algorithm
---------
The whole lexicographic objective is encoded as one *integer* edge weight
(with dominance constants derived from the instance bounds), so a single
maximum-weight bipartite matching (Hungarian method) solves the relaxed
problem where pairwise collisions and group atomicity are ignored.
Branch-and-bound then branches on every violated constraint of the relaxed
matching:

* a conflicting pair of allocations is split by requiring / forbidding one
  of the two edges;
* a split group (exactly one member matched) is split into "group out"
  (forbid every edge into either target) or one "group in via edge f"
  branch per edge that could allocate the missing member (edge required).

A node carries a forbidden-edge set *and* a required-edge set; the relaxed
matching fixes the required edges (collisions between them make the node
infeasible) and runs the Hungarian method on the residual bipartite graph.
Its value is an upper bound used for pruning.  Groups are therefore handled
jointly with single-target allocations -- the grouped optimum is never
derived by trimming an unconstrained optimum.  All geometry stays exact
(Fraction).
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .geometry import segment_distance_sq

TargetId = str
ArmId = str
Edge = Tuple[int, int]  # (arm index, target index)


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


@dataclass(frozen=True)
class FibreGroup:
    """A dual-fibre observation group: its two targets are atomic."""
    id: str
    target_a: TargetId
    target_b: TargetId


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
    reason: Optional[str] = None
    witness: Optional[dict] = None
    fibre_groups: Optional[List[dict]] = None


INF = 10**100


class Solver:
    def __init__(
        self,
        arms: Sequence[Arm],
        targets: Sequence[Target],
        clearance: int,
        minimum_allocations: int,
        groups: Sequence[FibreGroup] = (),
    ) -> None:
        self.arms = list(arms)
        self.targets = list(targets)
        self.clearance = clearance
        self.minimum_allocations = minimum_allocations
        self.minimum_requested = minimum_allocations
        self.n_arms = len(self.arms)
        self.n_targets = len(self.targets)
        self.clearance_sq = Fraction(clearance * clearance)

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

        # Dual-fibre groups kept in submission order.
        self.groups = list(groups)
        self.target_index: Dict[TargetId, int] = {
            t.id: j for j, t in enumerate(self.targets)
        }
        # group_of_target[j] is the group index owning target j, if any.
        self.group_of_target: List[Optional[int]] = [None] * self.n_targets
        self.group_targets: List[Tuple[int, int]] = []
        for gi, grp in enumerate(self.groups):
            ja, jb = (self.target_index[grp.target_a],
                      self.target_index[grp.target_b])
            self.group_of_target[ja] = gi
            self.group_of_target[jb] = gi
            self.group_targets.append((ja, jb))

        self._pair_cache: Dict[Tuple[int, int, int, int],
                               Tuple[bool, Fraction, dict]] = {}
        self._make_weights()

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

        self.weight: Dict[Edge, int] = {}
        for i, j in self.edges:
            stability_gain = (m - j) * place[i]  # vs. leaving arm i unused
            self.weight[(i, j)] = (
                self.c3
                + self.c2 * self.targets[j].priority
                - self.c1 * self.len2[i][j]
                + stability_gain
            )

    def matching_weight(self, assignment: Sequence[int]) -> int:
        return sum(self.weight[(i, j)] for i, j in enumerate(assignment) if j >= 0)

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
        for idx in range(len(used_edges)):
            for jdx in range(idx + 1, len(used_edges)):
                (i, j), (k, h) = used_edges[idx], used_edges[jdx]
                if self.pair_info(i, j, k, h)[0]:
                    return used_edges[idx], used_edges[jdx]
        return None

    def _branch_edge(self, chosen: List[Edge]) -> Edge:
        """Pick the chosen edge participating in the most internal clashes;
        ties broken by arm then target order."""
        degrees = {}
        chosen_sorted = sorted(chosen)
        for idx in range(len(chosen_sorted)):
            for jdx in range(idx + 1, len(chosen_sorted)):
                (i, j), (k, h) = chosen_sorted[idx], chosen_sorted[jdx]
                if self.pair_info(i, j, k, h)[0]:
                    degrees[(i, j)] = degrees.get((i, j), 0) + 1
                    degrees[(k, h)] = degrees.get((k, h), 0) + 1
        if not degrees:
            return chosen[0]
        return min(degrees, key=lambda e: (-degrees[e], e))

    # ------------------------------------------------------------------
    # Relaxed maximum-weight matching with forbidden / required edges
    # ------------------------------------------------------------------
    def _relaxed_match(
        self,
        forbidden: FrozenSet[Edge],
        required: FrozenSet[Edge],
    ) -> Optional[Tuple[int, List[Edge]]]:
        """Maximum-weight matching subject to edge availability.

        ``required`` edges are fixed into the matching; the result is
        ``None`` when they are unreachable, forbidden, share an arm/target
        or mutually violate clearance -- such a branch is infeasible.  The
        residual bipartite graph is solved by the Hungarian method.
        """
        n, m = self.n_arms, self.n_targets
        used_rows: Set[int] = set()
        used_cols: Set[int] = set()
        fixed: List[Edge] = []
        fixed_value = 0
        for edge in sorted(required):
            i, j = edge
            if (edge in forbidden or self.len2[i][j] is None
                    or i in used_rows or j in used_cols):
                return None
            used_rows.add(i)
            used_cols.add(j)
            fixed.append(edge)
            fixed_value += self.weight[edge]
        for idx in range(len(fixed)):
            for jdx in range(idx + 1, len(fixed)):
                (i, j), (k, h) = fixed[idx], fixed[jdx]
                a, b = (i, k) if i < k else (k, i)
                t1, t2 = (j, h) if i < k else (h, j)
                if self.pair_info(a, t1, b, t2)[0]:
                    return None

        free_rows = [i for i in range(n) if i not in used_rows]
        free_cols = [j for j in range(m) if j not in used_cols]
        chosen = fixed[:]
        total = fixed_value
        if free_rows and free_cols:
            size = max(len(free_rows), len(free_cols), 1)
            cost = [[0] * size for _ in range(size)]
            for ii, i in enumerate(free_rows):
                for jj, j in enumerate(free_cols):
                    if (self.len2[i][j] is not None
                            and (i, j) not in forbidden):
                        cost[ii][jj] = -self.weight[(i, j)]  # Hungarian min
            row_owner, _ = _hungarian(cost)
            for ii, i in enumerate(free_rows):
                jj = row_owner[ii]
                if 0 <= jj < len(free_cols):
                    j = free_cols[jj]
                    if (self.len2[i][j] is not None
                            and (i, j) not in forbidden):
                        chosen.append((i, j))
                        total += self.weight[(i, j)]
        return total, chosen

    # ------------------------------------------------------------------
    # Dual-fibre group helpers
    # ------------------------------------------------------------------
    def _split_group(self, chosen: List[Edge]
                     ) -> Optional[Tuple[int, int]]:
        """A group split by the relaxed matching: ``(group index, index of
        the member target left unmatched)``; ``None`` when every group is
        either fully in or fully out."""
        matched: Dict[int, int] = {}
        for i, j in chosen:
            matched[j] = i
        for gi, (ja, jb) in enumerate(self.group_targets):
            in_a, in_b = ja in matched, jb in matched
            if in_a != in_b:
                return gi, jb if in_a else ja
        return None

    def _conflicts(self, edge: Edge, others: Sequence[Edge]) -> bool:
        ei, ej = edge
        for k, h in others:
            if k == ei:
                continue
            a, b = (ei, k) if ei < k else (k, ei)
            t1, t2 = (ej, h) if ei < k else (h, ej)
            if self.pair_info(a, t1, b, t2)[0]:
                return True
        return False

    # ------------------------------------------------------------------
    # Branch and bound on collisions *and* group atomicity
    # ------------------------------------------------------------------
    def solve(self) -> Tuple[Tuple[int, ...], bool, Optional[dict]]:
        # Feasible incumbent from deterministic greedy seeds (enables pruning
        # at the root); then exact branching proves joint optimality over
        # whole groups and single targets.
        incumbent = self._greedy_seed()
        best_value = self.matching_weight(incumbent)
        best_assignment = incumbent[:]
        stats = {"nodes": 0}
        memo: Dict[Tuple[FrozenSet[Edge], FrozenSet[Edge]],
                  Optional[Tuple[int, List[Edge]]]] = {}

        def recurse(forbidden: FrozenSet[Edge],
                    required: FrozenSet[Edge]) -> None:
            nonlocal best_value, best_assignment
            stats["nodes"] += 1
            key = (forbidden, required)
            cached = memo.get(key)
            if cached is not None:
                relaxed = cached
            else:
                relaxed = self._relaxed_match(forbidden, required)
                memo[key] = relaxed
            if relaxed is None:
                return  # infeasible node
            bound, chosen = relaxed
            if bound <= best_value:
                return  # cannot improve the incumbent

            conflict = self._first_conflict(chosen)
            if conflict is None:
                split = self._split_group(chosen)
                if split is None:
                    # Relaxed optimum satisfies every hard constraint.
                    assignment = [-1] * self.n_arms
                    for i, j in chosen:
                        assignment[i] = j
                    best_value = bound
                    best_assignment = assignment
                    return
            else:
                split = None

            if conflict is not None:
                edge = self._branch_edge(chosen)
                ei, ej = edge
                # Branch 1: include ``edge`` -> require it, forbid every
                # other edge on its arm/target and every edge that collides
                # with it.  Infeasible if that contradicts an already
                # required edge.
                forced: set = set(forbidden)
                new_required: set = set(required)
                feasible_include = True
                for jj in range(self.n_targets):
                    if jj != ej and self.len2[ei][jj] is not None:
                        if (ei, jj) in required:
                            feasible_include = False
                        forced.add((ei, jj))
                for ii in range(self.n_arms):
                    if ii != ei and self.len2[ii][ej] is not None:
                        if (ii, ej) in required:
                            feasible_include = False
                        forced.add((ii, ej))
                for (k, h) in self.edges:
                    if k == ei:
                        continue
                    a, b = (ei, k) if ei < k else (k, ei)
                    t1, t2 = (ej, h) if ei < k else (h, ej)
                    if self.pair_info(a, t1, b, t2)[0]:
                        if (k, h) in required:
                            feasible_include = False
                        forced.add((k, h))
                if feasible_include:
                    new_required.add(edge)
                    recurse(frozenset(forced), frozenset(new_required))
                    if best_value == bound:
                        return  # reached the node's upper bound
                # Branch 2: exclude ``edge``.
                recurse(forbidden | {edge}, required)
                return

            # Group atomicity split: one member is matched, the other not.
            gi, missing = split  # type: ignore[misc]
            ja, jb = self.group_targets[gi]

            # Branch family A: the group is fully IN; the missing member is
            # allocated via one specific edge (arm -> missing target).
            busy_rows = {i for (i, _) in required}
            for i in range(self.n_arms):
                f = (i, missing)
                if (i in busy_rows or f in forbidden
                        or self.len2[i][missing] is None):
                    continue
                if self._conflicts(f, list(required)):
                    continue
                recurse(forbidden, required | {f})
                if best_value == bound:
                    return

            # Branch B: the group is fully OUT -> forbid every edge entering
            # either member target.  Infeasible if a required edge does so.
            out_forced = set(forbidden)
            feasible_out = True
            for j in (ja, jb):
                for ii in range(self.n_arms):
                    if self.len2[ii][j] is not None:
                        if (ii, j) in required:
                            feasible_out = False
                        out_forced.add((ii, j))
            if feasible_out:
                recurse(frozenset(out_forced), required)

        recurse(frozenset(), frozenset())
        self.solve_stats = stats

        best_tuple = tuple(best_assignment)
        count = sum(1 for x in best_tuple if x >= 0)
        witness = None
        self.atomic_summary = None
        if count < self.minimum_allocations:
            witness = self._shortfall_witness()
            if self.groups:
                self.atomic_summary = self._atomicity_summary(best_tuple)
                witness["dual_fibre"] = self.atomic_summary
        return best_tuple, count >= self.minimum_allocations, witness

    # ------------------------------------------------------------------
    # Greedy feasible seeds
    # ------------------------------------------------------------------
    def _greedy_seed(self) -> List[int]:
        """Best of several deterministic greedy, always-feasible passes.

        Every returned seed honours group atomicity: the plain passes are
        repaired by dropping the lone allocation of any split group, and an
        additional pass places whole groups first.
        """
        n, m = self.n_arms, self.n_targets
        candidates: List[List[int]] = []

        def greedy(arm_order, target_key):
            assignment = [-1] * n
            used = [False] * m
            placed: List[Edge] = []
            for i in arm_order:
                for j in sorted((j for j in range(m)
                                 if self.len2[i][j] is not None and not used[j]),
                                key=lambda j, i=i: target_key(i, j)):
                    clash = False
                    for k, h in placed:
                        a, b = (i, k) if i < k else (k, i)
                        t1, t2 = (j, h) if i < k else (h, j)
                        if self.pair_info(a, t1, b, t2)[0]:
                            clash = True
                            break
                    if not clash:
                        assignment[i] = j
                        used[j] = True
                        placed.append((i, j))
                        break
            return assignment

        def group_feasible(assignment: List[int]) -> bool:
            for ja, jb in self.group_targets:
                am = _arm_of(assignment, ja) >= 0
                bm = _arm_of(assignment, jb) >= 0
                if am != bm:
                    return False
            return True

        def repair(assignment: List[int]) -> List[int]:
            # Drop the lone matched member of every split group; iterate to
            # a fixed point (dropping never splits another group).
            assignment = assignment[:]
            for ja, jb in self.group_targets:
                ia, ib = _arm_of(assignment, ja), _arm_of(assignment, jb)
                if (ia >= 0) != (ib >= 0):
                    i = ia if ia >= 0 else ib
                    assignment[i] = -1
            return assignment

        candidates.append(repair(greedy(range(n), lambda i, j: j)))
        # Process the most constrained arms first.
        constrained = sorted(
            range(n),
            key=lambda i: (sum(1 for d2 in self.len2[i] if d2 is not None), i))
        candidates.append(repair(greedy(
            constrained, lambda i, j: -self.targets[j].priority)))
        candidates.append(repair(greedy(
            constrained, lambda i, j: self.len2[i][j])))
        if self.groups:
            candidates.append(repair(self._group_first_seed()))
        # Keep only genuinely feasible seeds.
        candidates = [c for c in candidates if group_feasible(c)]
        return max(candidates, key=self.matching_weight)

    def _group_first_seed(self) -> List[int]:
        """Greedy pass that allocates whole dual-fibre groups before any
        ungrouped target."""
        n = self.n_arms
        assignment = [-1] * n
        used = [False] * self.n_targets
        placed: List[Edge] = []

        def compatible(edge: Edge) -> bool:
            i, j = edge
            for k, h in placed:
                a, b = (i, k) if i < k else (k, i)
                t1, t2 = (j, h) if i < k else (h, j)
                if self.pair_info(a, t1, b, t2)[0]:
                    return False
            return True

        def place(edge: Edge) -> None:
            i, j = edge
            assignment[i] = j
            used[j] = True
            placed.append(edge)

        for ja, jb in self.group_targets:
            best_pair = None
            best_gain = None
            for i in range(n):
                if assignment[i] >= 0 or self.len2[i][ja] is None:
                    continue
                for k in range(n):
                    if k == i or assignment[k] >= 0 \
                            or self.len2[k][jb] is None:
                        continue
                    e1, e2 = (i, ja), (k, jb)
                    a, b = (i, k) if i < k else (k, i)
                    t1, t2 = (ja, jb) if i < k else (jb, ja)
                    if self.pair_info(a, t1, b, t2)[0]:
                        continue
                    if not compatible(e1) or not compatible(e2):
                        continue
                    gain = self.weight[e1] + self.weight[e2]
                    if best_gain is None or gain > best_gain:
                        best_gain, best_pair = gain, (e1, e2)
            if best_pair is not None:
                place(best_pair[0])
                place(best_pair[1])

        # Ungrouped targets fill any remaining arms, lowest index first.
        for i in range(n):
            if assignment[i] >= 0:
                continue
            for j in range(self.n_targets):
                if used[j] or self.len2[i][j] is None:
                    continue
                if self.group_of_target[j] is not None:
                    continue  # grouped targets are only taken as a pair
                if compatible((i, j)):
                    place((i, j))
                    break
        return assignment

    # ------------------------------------------------------------------
    # Shortfall explanation
    # ------------------------------------------------------------------
    def _shortfall_witness(self) -> dict:
        """Structural evidence when the minimum cannot be met: the tightest
        violating pair among individually reachable allocations (if any)."""
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
        return {
            "blocking_pair": blocking,
            "capacity_limit": min(self.n_arms, self.n_targets),
            "reachable_arm_target_links": self.total_reachable,
        }

    def _atomicity_summary(self, assignment: Sequence[int]) -> dict:
        """Evidence that the shortfall is driven (at least in part) by the
        dual-fibre atomicity constraint: compare the joint optimum with the
        optimum that would exist if groups were allowed to split.

        ``ungrouped_maximum`` is obtained with every group disabled; the
        number of split allocations in that optimum is the count of
        allocations forbidden by atomicity alone.
        """
        # Re-solve the same instance with no groups: the optimum without the
        # atomicity constraint (still reachability + uniqueness + clearance).
        relaxed = Solver(self.arms, self.targets, self.clearance,
                         self.minimum_allocations, groups=())
        relaxed_assignment, _, _ = relaxed.solve()
        ungrouped_maximum = sum(1 for x in relaxed_assignment if x >= 0)

        split_groups: List[dict] = []
        arm_of = {j: i for i, j in enumerate(relaxed_assignment) if j >= 0}
        for gi, (ja, jb) in enumerate(self.group_targets):
            grp = self.groups[gi]
            ia, ib = arm_of.get(ja), arm_of.get(jb)
            if (ia is not None) != (ib is not None):
                j = ja if ia is not None else jb
                i = ia if ia is not None else ib
                split_groups.append({
                    "group_id": grp.id,
                    "allocated_target": self.targets[j].id,
                    "allocated_arm": self.arms[i].id,
                    "unallocated_target": (
                        self.targets[jb].id if j == ja else self.targets[ja].id),
                })

        actual = sum(1 for x in assignment if x >= 0)
        atomic_cost = ungrouped_maximum - actual
        return {
            "constraint": "dual_fibre_atomic",
            "groups_submitted": len(self.groups),
            "maximum_without_atomicity": ungrouped_maximum,
            "maximum_with_atomicity": actual,
            "allocations_blocked_by_atomicity": atomic_cost,
            "would_be_split_groups": split_groups,
            "note": (
                "each dual-fibre group must be allocated as an indivisible "
                "pair (one arm per target) or not at all; allocations that "
                "would split a group are forbidden"
            ),
        }

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

        count = len(pairs)
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
        else:
            reason = None

        fibre_groups = None
        if self.groups:
            fibre_groups = self._group_records(assignment)
            if (not feasible and getattr(self, "atomic_summary", None)
                    is not None
                    and self.atomic_summary["allocations_blocked_by_atomicity"] > 0):
                blocked = self.atomic_summary["allocations_blocked_by_atomicity"]
                reason = (reason or "") + (
                    "; the dual-fibre atomic constraint forbids "
                    f"{blocked} allocation(s) that would otherwise be "
                    "available (each group's two targets may only be "
                    "allocated as an indivisible pair)"
                )

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
            minimum_requested=self.minimum_requested,
            feasible=feasible,
            reason=reason,
            witness=witness,
            fibre_groups=fibre_groups,
        )

    # ------------------------------------------------------------------
    # Per-group allocation records
    # ------------------------------------------------------------------
    def _group_records(self, assignment: Sequence[int]) -> List[dict]:
        """One record per submitted group: fully allocated (with the arm for
        each target) or wholly unallocated.  A partial record can never occur
        because the solver enforces atomicity."""
        records: List[dict] = []
        for gi, (ja, jb) in enumerate(self.group_targets):
            grp = self.groups[gi]
            ia, ib = _arm_of(assignment, ja), _arm_of(assignment, jb)
            status = "allocated" if ia >= 0 and ib >= 0 else "unallocated"
            records.append({
                "group_id": grp.id,
                "status": status,
                "members": [
                    self._member_record(grp.id, ia, ja, "a"),
                    self._member_record(grp.id, ib, jb, "b"),
                ],
            })
        return records

    def _member_record(self, group_id: str, arm_i: int, target_j: int,
                       slot: str) -> dict:
        record = {
            "group_id": group_id,
            "slot": slot,
            "target_id": self.targets[target_j].id,
            "allocated": arm_i >= 0,
            "arm_id": self.arms[arm_i].id if arm_i >= 0 else None,
            "extension": None,
            "extension_sq": None,
        }
        if arm_i >= 0:
            d2 = self.len2[arm_i][target_j] or 0
            record["extension"] = float(d2 ** 0.5)
            record["extension_sq"] = d2
        return record


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


def _arm_of(assignment: Sequence[int], target: int) -> int:
    """Index of the arm allocated to ``target`` in an assignment array, or
    -1 when the target is unused."""
    for i, j in enumerate(assignment):
        if j == target:
            return i
    return -1
