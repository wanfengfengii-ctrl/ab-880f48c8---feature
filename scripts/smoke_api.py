"""Collision-aware API smoke test, run from the one-shot ``verify`` service.

It waits for readiness and exercises:

1. a feasible 6-arm / 6-target adjudication, checking the returned pairings,
   per-arm lengths, uniqueness and *every* pairwise clearance against the
   requested integer clearance;
2. an instance where pairwise collisions make the requested minimum
   impossible: the API must return the maximum achievable count and a clear
   reason (HTTP 200, status minimum_not_met);
3. an illegal payload that must be rejected with HTTP 422 without solving;
4. dual-fibre groups: validation of disjoint target references, an atomic
   completion that jointly re-picks the assignment (never "old optimum minus
   the residual half"), and an atomicity-driven shortfall whose witness
   reports the ungrouped maximum and the allocations lost to atomicity.

Exits 0 on success, 1 on any failure.
"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
EPS = 1e-9


def request(method, path, payload=None, expect=(200,)):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            code, body = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        code, body = e.code, e.read()
    if code not in expect:
        raise AssertionError(f"{method} {path}: expected {expect}, got {code}: "
                             f"{body[:300]!r}")
    return code, json.loads(body)


def wait_ready(deadline=60.0):
    end = time.time() + deadline
    last = None
    while time.time() < end:
        try:
            code, body = request("GET", "/health/ready")
            if code == 200 and body.get("status") == "ready":
                return True
        except Exception as e:  # noqa: BLE001 - connection may not exist yet
            last = e
        time.sleep(1)
    raise AssertionError(f"API did not become ready in {deadline}s: {last}")


def feasible_payload():
    return {
        "arms": [{"id": f"A{i}", "x": i * 100, "y": 0, "max_extension": 500}
                 for i in range(6)],
        "targets": [{"id": f"T{i}", "x": i * 100 + 10, "y": 10,
                     "priority": i + 1} for i in range(6)],
        "clearance": 5,
        "minimum_allocations": 6,
    }


def collision_payload():
    # Two arms/targets per group; both pairings of a group have segments
    # closer than clearance 3 (converging or crossing), so at most one
    # allocation per group is possible -> maximum 3 of requested 6.
    arms, targets = [], []
    for g in range(3):
        gx = 200 * g
        arms += [{"id": f"L{g}", "x": gx, "y": 0, "max_extension": 100},
                 {"id": f"R{g}", "x": gx, "y": 10, "max_extension": 100}]
        targets += [{"id": f"lo{g}", "x": gx + 10, "y": 4, "priority": 5 + g},
                    {"id": f"hi{g}", "x": gx + 10, "y": 6, "priority": 1 + g}]
    return {"arms": arms, "targets": targets,
            "clearance": 3, "minimum_allocations": 6}


def group_completion_payload():
    # Local arms A0/A1; targets a (A0-only, high priority), b (A1-only, low),
    # c (both arms, high priority).  Ungrouped optimum takes a+c; the
    # dual-fibre group (a,b) must instead complete a+b jointly, dropping c
    # while keeping the full allocation count.
    arms = [{"id": "A0", "x": 0, "y": 0, "max_extension": 15},
            {"id": "A1", "x": 10, "y": 0, "max_extension": 15}] + \
           [{"id": f"A{i}", "x": i * 1000, "y": 0, "max_extension": 30}
            for i in range(2, 6)]
    targets = [
        {"id": "a", "x": -10, "y": 0, "priority": 9},
        {"id": "b", "x": 20, "y": 0, "priority": 1},
        {"id": "c", "x": 5, "y": 0, "priority": 9},
    ] + [{"id": f"F{i}", "x": i * 1000 + 2, "y": 0, "priority": 1}
         for i in range(2, 6)]
    return {"arms": arms, "targets": targets, "clearance": 1,
            "minimum_allocations": 6,
            "dual_fibre_groups": [{"target_a": "a", "target_b": "b"}]}


def group_shortfall_payload():
    # Two arms share a base (always clash); with targets T0/T1 grouped the
    # group cannot complete and neither target may be taken alone -> joint
    # maximum 4, while the ungrouped maximum is 5.
    arms = [{"id": "A0", "x": 0, "y": 0, "max_extension": 200},
            {"id": "A1", "x": 0, "y": 0, "max_extension": 200}] + \
           [{"id": f"A{i}", "x": i * 1000, "y": 0, "max_extension": 200}
            for i in range(2, 6)]
    targets = [{"id": "T0", "x": 10, "y": 0, "priority": 5},
               {"id": "T1", "x": 12, "y": 0, "priority": 9}] + \
              [{"id": f"F{j}", "x": j * 1000 + 10, "y": 0, "priority": 5}
               for j in range(2, 6)]
    return {"arms": arms, "targets": targets, "clearance": 1,
            "minimum_allocations": 6,
            "dual_fibre_groups": [{"target_a": "T0", "target_b": "T1"}]}


def main():
    wait_ready()
    print("readiness OK")

    # 1) feasible instance + evidence verification
    _, body = request("POST", "/api/v1/adjudicate", feasible_payload())
    assert body["status"] == "ok" and body["feasible"] is True, body
    obj = body["objective"]
    assert obj["num_allocations"] == 6, obj
    assert obj["priority_sum"] == 21, obj

    arms_used = [p["arm_id"] for p in body["assignments"]]
    tgts_used = [p["target_id"] for p in body["assignments"]]
    assert len(set(arms_used)) == len(arms_used) == 6
    assert len(set(tgts_used)) == len(tgts_used) == 6
    assert body["unassigned_arms"] == [] and body["unassigned_targets"] == []

    # per-arm lengths present and consistent
    for p in body["assignments"]:
        length = body["arm_lengths"][p["arm_id"]]
        assert math.isclose(length, p["extension"], abs_tol=1e-12)
        assert length > 0

    # every pair of closed segments respects the integer clearance
    evidence = body["clearance"]["pair_evidence"]
    assert len(evidence) == math.comb(6, 2) == 15
    for rec in evidence:
        assert rec["satisfied"] is True, rec
        assert rec["distance"] + EPS >= rec["required_clearance"], rec
        assert rec["witness"]["point_on_first"]
    assert body["clearance"]["all_pairs_satisfied"] is True
    print("feasible case OK: 6 pairs, 15 clearances verified")

    # 2) collision-driven shortfall
    _, body = request("POST", "/api/v1/adjudicate", collision_payload())
    assert body["status"] == "minimum_not_met", body
    assert body["feasible"] is False
    assert body["objective"]["num_allocations"] == 3, body["objective"]
    assert "minimum_allocations=6" in body["reason"]
    bp = body["witness"]["blocking_pair"]
    assert bp is not None and bp["distance"] < bp["required_clearance"]
    for rec in body["clearance"]["pair_evidence"]:
        assert rec["satisfied"] is True
        assert rec["distance"] + EPS >= rec["required_clearance"]
    print("collision shortfall OK: max=3 reported with reason and witness")

    # 3) illegal input rejected before solving
    bad = feasible_payload()
    bad["clearance"] = 0
    request("POST", "/api/v1/adjudicate", bad, expect=(422,))
    bad = feasible_payload()
    bad["arms"] = bad["arms"][:5]
    request("POST", "/api/v1/adjudicate", bad, expect=(422,))
    print("illegal input rejected with 422")

    # 4a) group validation: self-pair, unknown target, reused target
    bad = feasible_payload()
    bad["dual_fibre_groups"] = [{"target_a": "T0", "target_b": "T0"}]
    request("POST", "/api/v1/adjudicate", bad, expect=(422,))
    bad = feasible_payload()
    bad["dual_fibre_groups"] = [{"target_a": "T0", "target_b": "Nope"}]
    request("POST", "/api/v1/adjudicate", bad, expect=(422,))
    bad = feasible_payload()
    bad["dual_fibre_groups"] = [
        {"target_a": "T0", "target_b": "T1"},
        {"target_a": "T1", "target_b": "T2"},
    ]
    request("POST", "/api/v1/adjudicate", bad, expect=(422,))
    print("illegal groups rejected with 422")

    # 4b) joint completion, not residual deletion
    _, body = request("POST", "/api/v1/adjudicate", group_completion_payload())
    assert body["status"] == "ok" and body["feasible"] is True, body
    assert body["objective"]["num_allocations"] == 6, body["objective"]
    by_target = {p["target_id"]: p["arm_id"] for p in body["assignments"]}
    assert by_target["a"] == "A0" and by_target["b"] == "A1", by_target
    assert "c" not in by_target, by_target
    g = body["dual_fibre_groups"][0]
    assert g["allocated"] is True and len(g["allocations"]) == 2
    assert len({a["arm_id"] for a in g["allocations"]}) == 2
    print("group completion OK: jointly re-picked (a+b, c dropped), count=6")

    # 4c) atomicity-driven shortfall with explicit attribution
    _, body = request("POST", "/api/v1/adjudicate", group_shortfall_payload())
    assert body["status"] == "minimum_not_met", body
    assert body["objective"]["num_allocations"] == 4, body["objective"]
    g = body["dual_fibre_groups"][0]
    assert g["allocated"] is False and g["allocations"] == []
    d = body["witness"]["dual_fibre_shortfall"]
    assert d["grouped_max_allocations"] == 4, d
    assert d["ungrouped_max_allocations"] == 5, d
    assert d["allocations_lost_to_atomicity"] == 1, d
    assert "all-or-nothing group constraint" in body["reason"]
    for rec in body["clearance"]["pair_evidence"]:
        assert rec["satisfied"] is True
    # same instance without groups reports the ungrouped maximum 5
    plain = group_shortfall_payload()
    plain.pop("dual_fibre_groups")
    _, body = request("POST", "/api/v1/adjudicate", plain)
    assert body["objective"]["num_allocations"] == 5, body["objective"]
    print("group shortfall OK: joint max=4 vs ungrouped max=5, attributed")

    print("SMOKE: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"SMOKE: FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"SMOKE: ERROR: {e!r}", file=sys.stderr)
        sys.exit(1)
