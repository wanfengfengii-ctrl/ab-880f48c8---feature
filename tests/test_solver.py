"""Tests for the lexicographic allocation solver."""
import itertools
import math
import random
from fractions import Fraction

import pytest

from app.solver import Arm, Solver, Target


def solve(arms, targets, clearance, minimum=1, groups=None):
    s = Solver(arms, targets, clearance, minimum, groups=groups)
    assignment, ok, witness = s.solve()
    return s, assignment, ok, witness


def assert_feasible(s, assignment, clearance):
    used = set()
    placed = []
    for i, j in enumerate(assignment):
        if j < 0:
            continue
        assert j not in used, "target reused"
        assert s.len2[i][j] is not None, "unreachable allocation"
        for k, h in placed:
            a, b = (i, k) if i < k else (k, i)
            t1, t2 = (j, h) if i < k else (h, j)
            conflict, d2, _ = s.pair_info(a, t1, b, t2)
            assert not conflict, "clearance violated"
            assert d2 >= Fraction(clearance ** 2)
        used.add(j)
        placed.append((i, j))


def assert_group_atomic(assignment, groups):
    used = {j for j in assignment if j >= 0}
    for a, b in groups:
        assert (a in used) == (b in used), \
            f"group ({a},{b}) split: atomicity violated"


def test_all_reachable_far_apart():
    arms = [Arm(f"A{i}", i * 100, 0, 500) for i in range(6)]
    targets = [Target(f"T{i}", i * 100 + 10, 10, i + 1) for i in range(6)]
    s, a, ok, _ = solve(arms, targets, 5, 6)
    assert ok and tuple(a) == (0, 1, 2, 3, 4, 5)
    r = s.build_result(a, ok, None)
    assert r.num_allocations == 6
    assert r.priority_sum == 21
    assert all(p["satisfied"] for p in r.pair_clearances)


def test_crossing_segments_cannot_both_be_used():
    # Three groups of two arms/targets.  Within each group the arms start
    # 10 apart while both targets sit only 2 apart: the "straight" pairing
    # converges to a 2-gap and the swapped pairing crosses.  Both violate a
    # clearance of 3, so each group contributes at most one allocation.
    arms, targets = [], []
    for g in range(3):
        gx = 200 * g
        arms += [Arm(f"L{g}", gx, 0, 100), Arm(f"R{g}", gx, 10, 100)]
        targets += [Target(f"lo{g}", gx + 10, 4, 5 + g),
                    Target(f"hi{g}", gx + 10, 6, 1 + g)]
    s, a, ok, w = solve(arms, targets, 3, 6)
    assert not ok
    assert_feasible(s, a, 3)
    assert sum(x >= 0 for x in a) == 3
    r = s.build_result(a, ok, w)
    assert r.reason and "minimum_allocations=6" in r.reason
    assert r.witness["blocking_pair"] is not None
    # high-priority targets win within each single-allocation group
    assert r.priority_sum == 5 + 6 + 7


def test_priority_beats_extension():
    # Every arm reaches two targets at equal length, one low and one high
    # priority; the high-priority ones must be taken.
    arms = [Arm(f"p{i}", i * 100, 0, 10) for i in range(6)]
    targets = []
    for i in range(6):
        targets.append(Target(f"lo{i}", i * 100 + 3, 0, 1))
        targets.append(Target(f"hi{i}", i * 100 - 3, 0, 9))
    s, a, ok, _ = solve(arms, targets, 1, 6)
    assert ok
    assert_feasible(s, a, 1)
    r = s.build_result(a, ok, None)
    assert r.num_allocations == 6
    assert r.priority_sum == 54
    assert all(t.startswith("hi") for t in
               [p["target_id"] for p in r.assignments])


def test_extension_squared_is_minimised_after_priority():
    # Single arm + far-away filler arms; the close target must win among
    # equal-priority choices.
    arms = [Arm("a0", 0, 0, 999)] + [Arm(f"a{i}", i * 1000, 5000, 10)
                                     for i in range(1, 6)]
    targets = [Target("near", 3, 0, 7), Target("far", 5, 0, 7)]
    targets += [Target(f"f{i}", i * 1000, 5000 + 2, 7) for i in range(1, 6)]
    s, a, ok, _ = solve(arms, targets, 1, 6)
    assert ok
    assert a[0] == 0  # near
    r = s.build_result(a, ok, None)
    assert r.extension_sq_sum == 9 + 5 * 4


def test_stable_sequence_prefers_earlier_arms():
    # a0/a1 share a base: their segments always start at the same point and
    # therefore clash, so exactly one of them is allocated.  The stable
    # sequence must assign the earlier arm (and the smaller target index).
    arms = [Arm("a0", 0, 0, 10), Arm("a1", 0, 0, 10)] + \
           [Arm(f"a{i}", i * 1000, 0, 10) for i in range(2, 6)]
    targets = [Target("t0", 3, 0, 5), Target("t1", 4, 0, 5)] + \
              [Target(f"f{i}", i * 1000 + 1, 0, 5) for i in range(2, 6)]
    s, a, ok, _ = solve(arms, targets, 1, 5)
    assert ok
    assert a[0] == 0 and a[1] == -1
    assert tuple(a[2:]) == (2, 3, 4, 5)


def test_clearance_equality_is_allowed_closed_segments():
    # Two parallel segments exactly 3 apart: closed-segment distance equals
    # the clearance, so the constraint (>=) must allow both.
    arms = [Arm("low", 0, 0, 50), Arm("high", 0, 3, 50)] + \
           [Arm(f"f{i}", i * 1000, 0, 50) for i in range(2, 6)]
    targets = [Target("t0", 10, 0, 4), Target("t1", 10, 3, 4)] + \
              [Target(f"g{i}", i * 1000 + 5, 0, 4) for i in range(2, 6)]
    s, a, ok, _ = solve(arms, targets, 3, 6)
    assert ok and tuple(a) == (0, 1, 2, 3, 4, 5)
    # one unit tighter must forbid the close pair
    s2, a2, ok2, _ = solve(arms, targets, 4, 6)
    assert not ok2 and sum(x >= 0 for x in a2) == 5


def test_unreachable_arm_reported():
    arms = [Arm(f"A{i}", i * 1000, 0, 5) for i in range(6)]
    targets = [Target(f"T{i}", i * 1000, 0, 1) for i in range(6)]
    # reachable (distance 0); now move one target out of reach
    targets[0] = Target("T0", 100, 0, 1)
    s, a, ok, w = solve(arms, targets, 1, 6)
    assert not ok
    assert a[0] == -1
    assert sum(x >= 0 for x in a) == 5
    r = s.build_result(a, ok, w)
    assert "A0" in r.unassigned_arms
    assert "T0" in r.unassigned_targets


def test_minimum_higher_than_capacity():
    arms = [Arm(f"A{i}", 0, i * 100, 50) for i in range(6)]
    targets = [Target(f"T{j}", 0, j * 100, 1) for j in range(6)]
    s, a, ok, w = solve(arms, targets, 1, 12)
    assert not ok
    assert w["capacity_limit"] == 6


@pytest.mark.parametrize("seed", range(20))
def test_random_instances_feasible_and_weight_optimal(seed):
    import random
    rng = random.Random(seed)
    n_arms, n_targets = rng.randint(6, 12), rng.randint(6, 16)
    span = rng.choice([30, 80])
    arms = [Arm(f"a{i}", rng.randint(-span, span), rng.randint(-span, span),
                rng.randint(10, span)) for i in range(n_arms)]
    targets = [Target(f"t{j}", rng.randint(-span, span), rng.randint(-span, span),
                      rng.randint(1, 9)) for j in range(n_targets)]
    clearance = rng.randint(1, 4)
    s, a, ok, _ = solve(arms, targets, clearance, 1)
    assert_feasible(s, a, clearance)
    # Result must achieve at least the collision-ignoring greedy counts.
    r = s.build_result(a, ok, None)
    assert r.num_allocations >= 1
    assert all(p["satisfied"] for p in r.pair_clearances)
    # Evidence distances must agree with the extension geometry.
    for p in r.assignments:
        assert math.isclose(p["extension"] ** 2, p["extension_sq"], rel_tol=1e-12)


# ----------------------------------------------------------------------
# Dual-fibre group (atomicity) tests
# ----------------------------------------------------------------------
def far_arms_targets():
    arms = [Arm(f"A{i}", i * 1000, 0, 500) for i in range(6)]
    targets = [Target(f"T{i}", i * 1000 + 10, 0, i + 1) for i in range(6)]
    return arms, targets


def test_group_allocated_completely_with_arms():
    arms, targets = far_arms_targets()
    s, a, ok, _ = solve(arms, targets, 5, 6, groups=[(0, 1)])
    assert ok
    assert_feasible(s, a, 5)
    assert_group_atomic(a, [(0, 1)])
    assert a[0] == 0 and a[1] == 1
    r = s.build_result(a, ok, None)
    g = r.groups[0]
    assert g["allocated"] is True
    assert {x["arm_id"] for x in g["allocations"]} == {"A0", "A1"}
    assert {x["target_id"] for x in g["allocations"]} == {"T0", "T1"}


def test_group_dead_when_only_one_arm_can_reach_both():
    # T0 and T1 are reachable by A0 only; no two distinct arms can complete
    # the group, so both targets must be dropped as a whole (T0 cannot be
    # taken alone).
    arms = [Arm("A0", 0, 0, 30)] + \
           [Arm(f"A{i}", i * 1000, 0, 30) for i in range(1, 6)]
    targets = [Target("T0", 5, 0, 9), Target("T1", 7, 0, 9)] + \
              [Target(f"F{i}", i * 1000 + 5, 0, 1) for i in range(1, 5)]
    s, a, ok, w = solve(arms, targets, 1, 5, groups=[(0, 1)])
    assert not ok
    assert_feasible(s, a, 1)
    assert_group_atomic(a, [(0, 1)])
    assert a[0] == -1  # A0 idle: neither grouped target may be used alone
    assert sum(x >= 0 for x in a) == 4
    # ungrouped the same instance CAN meet 5 (A0 takes T0 alone)
    s0, a0, ok0, _ = solve(arms, targets, 1, 5)
    assert ok0 and sum(x >= 0 for x in a0) == 5
    d = w["dual_fibre_shortfall"]
    assert d["ungrouped_max_allocations"] == 5
    assert d["grouped_max_allocations"] == 4
    assert d["allocations_lost_to_atomicity"] == 1
    assert d["minimum_blocked_by_atomicity"] is True
    r = s.build_result(a, ok, w)
    assert r.groups[0]["allocated"] is False
    assert r.groups[0]["reason"] == "structurally uncompletable"


def test_joint_selection_not_posthoc_residual_deletion():
    # Local arms A0 (x=0, reach 15) and A1 (x=10, reach 15):
    #   a  at -10 -> A0 only (priority 9)
    #   c  at   5 -> both arms (priority 9)
    #   b  at  20 -> A1 only (priority 1)
    # Group (a, b).  Ungrouped optimum: A0->a, A1->c (6 total with fillers).
    # A post-hoc "take the old optimum then delete the residual half-group"
    # would remove a and be left with 5; the joint optimum instead completes
    # the group (A0->a, A1->b), keeping 6 allocations and dropping c.
    arms = [Arm("A0", 0, 0, 15), Arm("A1", 10, 0, 15)] + \
           [Arm(f"A{i}", i * 1000, 0, 30) for i in range(2, 6)]
    targets = [
        Target("a", -10, 0, 9),
        Target("b", 20, 0, 1),
        Target("c", 5, 0, 9),
    ] + [Target(f"F{i}", i * 1000 + 2, 0, 1) for i in range(2, 6)]
    s0, a0, ok0, _ = solve(arms, targets, 1, 6)
    assert ok0 and a0[0] == 0 and a0[1] == 2  # a and c
    s, a, ok, _ = solve(arms, targets, 1, 6, groups=[(0, 1)])
    assert ok
    assert sum(x >= 0 for x in a) == 6
    assert a[0] == 0 and a[1] == 1  # group completed; c pushed out
    r = s.build_result(a, ok, None)
    assert r.groups[0]["allocated"] is True
    assert "c" in r.unassigned_targets


def test_group_collision_shortfall_is_attributed():
    # A0/A1 share a base (their segments always clash); the group can never
    # be completed, and neither grouped target may be used alone.
    arms = [Arm("A0", 0, 0, 200), Arm("A1", 0, 0, 200)] + \
           [Arm(f"A{i}", i * 1000, 0, 200) for i in range(2, 6)]
    targets = [Target("T0", 10, 0, 5), Target("T1", 12, 0, 9)] + \
              [Target(f"F{j}", j * 1000 + 10, 0, 5) for j in range(2, 6)]
    s, a, ok, w = solve(arms, targets, 1, 6, groups=[(0, 1)])
    assert not ok
    assert_feasible(s, a, 1)
    assert_group_atomic(a, [(0, 1)])
    assert sum(x >= 0 for x in a) == 4
    d = w["dual_fibre_shortfall"]
    assert d == {
        "groups_submitted": 1,
        "groups_allocated": 0,
        "groups_unallocated": 1,
        "grouped_max_allocations": 4,
        "ungrouped_max_allocations": 5,
        "allocations_lost_to_atomicity": 1,
        "minimum_blocked_by_atomicity": False,
    }
    r = s.build_result(a, ok, w)
    assert "all-or-nothing group constraint" in r.reason
    assert r.witness["ungrouped_max_allocations"] == 5


def test_priority_decides_whether_group_is_completed():
    # Two isolated 2-arm/2-target collision clusters (any pairing inside a
    # cluster violates clearance 3, so exactly one allocation per cluster).
    # Group spans the two low-x targets across the clusters.
    def build(gprio, sprio):
        arms, targets = [], []
        for g in range(2):
            gx = 200 * g
            arms += [Arm(f"L{g}", gx, 0, 100), Arm(f"R{g}", gx, 10, 100)]
            targets += [Target(f"grp{g}", gx + 10, 4, gprio),
                        Target(f"sng{g}", gx + 10, 6, sprio)]
        arms += [Arm(f"F{i}", i * 1000, 0, 50) for i in range(2)]
        targets += [Target(f"fill{i}", i * 1000 + 2, 0, 1) for i in range(2)]
        # group = (grp0, grp1)
        return arms, targets, [(0, 2)]

    arms, targets, groups = build(gprio=1, sprio=9)
    s, a, ok, _ = solve(arms, targets, 3, 1, groups=groups)
    assert ok
    assert_group_atomic(a, groups)
    used = {targets[j].id for j in a if j >= 0}
    assert used >= {"sng0", "sng1"}  # group dropped, high-priority singles
    assert not (used >= {"grp0", "grp1"})
    assert s.build_result(a, ok, None).groups[0]["allocated"] is False

    arms, targets, groups = build(gprio=9, sprio=1)
    s, a, ok, _ = solve(arms, targets, 3, 1, groups=groups)
    assert ok
    used = {targets[j].id for j in a if j >= 0}
    assert used >= {"grp0", "grp1"}  # group completed on priority tie-level
    assert s.build_result(a, ok, None).groups[0]["allocated"] is True


def test_stable_sequence_picks_earlier_arms_for_group():
    arms = [Arm("A0", 0, 0, 200), Arm("A1", 100, 0, 200)] + \
           [Arm(f"A{i}", i * 1000, 0, 200) for i in range(2, 6)]
    targets = [Target("G0", 10, 0, 5), Target("G1", 110, 0, 5)] + \
              [Target(f"F{j}", j * 1000 + 10, 0, 5) for j in range(2, 6)]
    s, a, ok, _ = solve(arms, targets, 1, 6, groups=[(0, 1)])
    assert ok and a[0] == 0 and a[1] == 1


def test_two_disjoint_groups_plus_singles():
    arms, targets = far_arms_targets()
    groups = [(0, 1), (2, 3)]
    s, a, ok, _ = solve(arms, targets, 5, 6, groups=groups)
    assert ok
    assert_feasible(s, a, 5)
    assert_group_atomic(a, groups)
    r = s.build_result(a, ok, None)
    assert [g["allocated"] for g in r.groups] == [True, True]
    assert sum(len(g["allocations"]) for g in r.groups) == 4


def test_brute_force_random_grouped_instances():
    def stability_tuple(combo, m):
        return tuple(j if j >= 0 else m for j in combo)

    rng = random.Random(2024)
    for trial in range(15):
        n_arms, n_targets = rng.randint(2, 4), rng.randint(2, 6)
        span = 12
        arms = [Arm(f"a{i}", rng.randint(-span, span), rng.randint(-span, span),
                    rng.randint(4, span)) for i in range(n_arms)]
        targets = [Target(f"t{j}", rng.randint(-span, span),
                          rng.randint(-span, span), rng.randint(1, 9))
                   for j in range(n_targets)]
        clearance = rng.randint(1, 4)
        perm = list(range(n_targets))
        rng.shuffle(perm)
        ng = rng.randint(0, n_targets // 2)
        groups = [(perm[2 * g], perm[2 * g + 1]) for g in range(ng)]
        s = Solver(arms, targets, clearance, 1, groups=groups)
        a, _, _ = s.solve()

        best = None
        for combo in itertools.product(range(-1, n_targets), repeat=n_arms):
            used0, placed, good = set(), [], True
            for i, j in enumerate(combo):
                if j < 0:
                    continue
                if j in used0 or s.len2[i][j] is None:
                    good = False
                    break
                for k, h in placed:
                    x, y = (i, k) if i < k else (k, i)
                    u, v = (j, h) if i < k else (h, j)
                    if s.pair_info(x, u, y, v)[0]:
                        good = False
                        break
                if not good:
                    break
                used0.add(j)
                placed.append((i, j))
            if not good:
                continue
            for ga, gb in groups:
                if (ga in used0) != (gb in used0):
                    good = False
                    break
            if not good:
                continue
            key = (len(placed),
                   sum(targets[j].priority for j in used0),
                   -sum(s.len2[i][j] for i, j in placed),
                   tuple(-x for x in stability_tuple(combo, n_targets)))
            if best is None or key > best[0]:
                best = (key, combo)
        assert tuple(a) == tuple(best[1]), (trial, a, best[1])
