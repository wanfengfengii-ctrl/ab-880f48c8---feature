"""Tests for dual-fibre observation groups (atomic pair allocation)."""
from fractions import Fraction

from app.solver import Arm, FibreGroup, Solver, Target


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


def assert_groups_atomic(assignment, group_targets):
    for ja, jb in group_targets:
        members = [j for j in assignment if j in (ja, jb)]
        assert len(members) in (0, 2), f"group split: {members}"


def test_group_is_allocated_when_it_does_not_cost_count():
    # Six arms far apart; group members sit under A0/A1.  Taking the whole
    # group is optimal (count 6), with one arm per member.
    arms = [Arm(f"A{i}", i * 100, 0, 500) for i in range(6)]
    targets = [Target(f"T{i}", i * 100 + 10, 10, i + 1) for i in range(6)]
    groups = [FibreGroup("G0", "T0", "T1")]
    s = Solver(arms, targets, 5, 6, groups=groups)
    a, ok, _ = s.solve()
    assert ok and tuple(a) == (0, 1, 2, 3, 4, 5)
    assert_feasible(s, a, 5)
    assert_groups_atomic(a, [(0, 1)])
    r = s.build_result(a, ok, None)
    rec = {g["group_id"]: g for g in r.fibre_groups}["G0"]
    assert rec["status"] == "allocated"
    arms_used = {m["target_id"]: m["arm_id"] for m in rec["members"]}
    assert arms_used == {"T0": "A0", "T1": "A1"}
    assert all(m["extension_sq"] == 200 for m in rec["members"])


def test_group_kept_whole_rather_than_trimming_old_optimum():
    # Without groups the optimum is A0->t0, A1->s1, A2->s2 (count 3, prio 25):
    # t0 (priority 9) squeezes s1 out.  Naively deleting the broken group
    # would leave count 2.  The joint optimum must keep the WHOLE group and
    # still reach count 3: A0->t0, A1->t1, A2->s2.
    arms = [Arm("A0", 0, 0, 10), Arm("A1", 100, 0, 100),
            Arm("A2", 300, 0, 10)]
    targets = [
        Target("t0", 5, 0, 9),    # A0 only
        Target("t1", 95, 0, 1),   # A1 only
        Target("s1", 105, 0, 8),  # A1 only (same arm as t1)
        Target("s2", 305, 0, 8),  # A2 only
    ]
    groups = [FibreGroup("G1", "t0", "t1")]
    s = Solver(arms, targets, 1, 3, groups=groups)
    a, ok, _ = s.solve()
    assert ok and tuple(a) == (0, 1, 3)
    assert_feasible(s, a, 1)
    assert_groups_atomic(a, [(0, 1)])
    r = s.build_result(a, ok, None)
    assert r.num_allocations == 3 and r.priority_sum == 18

    # The ungrouped optimum really does prefer splitting, proving the joint
    # solver is not just post-editing it.
    s0 = Solver(arms, targets, 1, 3, groups=())
    a0, ok0, _ = s0.solve()
    assert tuple(a0) == (0, 2, 3) and ok0


def test_group_dropped_whole_when_pair_incompatible():
    # Both group members are only reachable by the same arm, so the group can
    # never be allocated as a pair: it must be dropped wholesale, not split.
    arms = [Arm("A0", 0, 0, 100), Arm("A1", 300, 0, 10),
            Arm("A2", 400, 0, 10)]
    targets = [
        Target("t0", 5, 0, 9), Target("t1", 10, 0, 9),
        Target("s1", 305, 0, 1), Target("s2", 405, 0, 1),
    ]
    groups = [FibreGroup("G1", "t0", "t1")]
    s = Solver(arms, targets, 1, 3, groups=groups)
    a, ok, _ = s.solve()
    assert tuple(a) == (-1, 2, 3)
    assert_groups_atomic(a, [(0, 1)])
    assert not ok  # only 2 < requested 3
    r = s.build_result(a, ok, s._shortfall_witness())
    rec = r.fibre_groups[0]
    assert rec["status"] == "unallocated"
    assert all(m["allocated"] is False and m["arm_id"] is None
               for m in rec["members"])


def test_group_skipped_when_its_two_segments_collide():
    # Both members are reachable, but every simultaneous placement violates
    # clearance, so the group must be fully out -- not even one member may be
    # taken, since that would split the atomic pair.
    arms = [Arm("lo", 0, 0, 100), Arm("hi", 0, 10, 100)]
    targets = [Target("c0", 10, 4, 5), Target("c1", 10, 6, 5)]
    groups = [FibreGroup("G1", "c0", "c1")]
    s = Solver(arms, targets, 3, 1, groups=groups)
    a, ok, w = s.solve()
    assert tuple(a) == (-1, -1)  # the whole group is excluded
    assert not ok and w is not None
    assert_groups_atomic(a, [(0, 1)])

    # Without the group the same geometry admits exactly one allocation,
    # which meets minimum_allocations=1.
    s0 = Solver(arms, targets, 3, 1, groups=())
    a0, ok0, _ = s0.solve()
    assert sum(x >= 0 for x in a0) == 1 and ok0


def test_two_groups_and_singles_are_chosen_jointly():
    # 4 arms, two atomic pairs with unique reachable arms; both pairs must be
    # selected together (count 4).
    arms = [Arm(f"A{i}", i * 100, 0, 50) for i in range(4)]
    targets = [
        Target("g0a", 5, 0, 9), Target("g0b", 105, 0, 9),
        Target("g1a", 205, 0, 9), Target("g1b", 305, 0, 9),
    ]
    groups = [FibreGroup("G0", "g0a", "g0b"),
              FibreGroup("G1", "g1a", "g1b")]
    s = Solver(arms, targets, 1, 4, groups=groups)
    a, ok, _ = s.solve()
    assert ok and tuple(a) == (0, 1, 2, 3)
    assert_groups_atomic(a, [(0, 1), (2, 3)])


def test_shortfall_witness_names_atomicity():
    arms = [Arm("A0", 0, 0, 100), Arm("A1", 300, 0, 10),
            Arm("A2", 400, 0, 10)]
    targets = [
        Target("t0", 5, 0, 9), Target("t1", 10, 0, 9),
        Target("s1", 305, 0, 1), Target("s2", 405, 0, 1),
    ]
    groups = [FibreGroup("G1", "t0", "t1")]
    s = Solver(arms, targets, 1, 4, groups=groups)
    a, ok, w = s.solve()
    assert not ok
    assert w["dual_fibre"]["constraint"] == "dual_fibre_atomic"
    assert w["dual_fibre"]["maximum_without_atomicity"] == 3
    assert w["dual_fibre"]["maximum_with_atomicity"] == 2
    assert w["dual_fibre"]["allocations_blocked_by_atomicity"] == 1
    split = w["dual_fibre"]["would_be_split_groups"]
    assert len(split) == 1 and split[0]["group_id"] == "G1"
    r = s.build_result(a, ok, w)
    assert "atomic" in r.reason and "1 allocation" in r.reason


def test_ungrouped_targets_still_allocated_alongside_groups():
    # A dropped group must not prevent unrelated singles from being taken.
    arms = [Arm("A0", 0, 0, 10), Arm("A1", 1000, 0, 10),
            Arm("A2", 2000, 0, 10)]
    targets = [
        Target("t0", 5, 0, 9), Target("t1", 10, 0, 9),  # group infeasible pair
        Target("u1", 1005, 0, 7), Target("u2", 2005, 0, 7),
    ]
    groups = [FibreGroup("G1", "t0", "t1")]
    s = Solver(arms, targets, 1, 2, groups=groups)
    a, ok, _ = s.solve()
    assert ok and tuple(a) == (-1, 2, 3)
    assert_groups_atomic(a, [(0, 1)])
