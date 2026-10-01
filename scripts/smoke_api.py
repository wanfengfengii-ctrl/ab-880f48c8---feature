"""Collision-aware API smoke test, run from the one-shot ``verify`` service.

It waits for readiness and exercises:

1. a feasible 6-arm / 6-target adjudication, checking the returned pairings,
   per-arm lengths, uniqueness and *every* pairwise clearance against the
   requested integer clearance;
2. an instance where pairwise collisions make the requested minimum
   impossible: the API must return the maximum achievable count and a clear
   reason (HTTP 200, status minimum_not_met);
3. an illegal payload that must be rejected with HTTP 422 without solving;
4. dual-fibre observation groups: a group allocated as one indivisible pair
   with one arm per member, a group the joint optimum must keep whole rather
   than trimming an ungrouped optimum, and an impossible group whose
   atomicity causes a reported shortfall.

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


def group_kept_payload():
    # Ungrouped optimum splits group G1=(t0,t1): A0->t0, A1->s1, ... and
    # naive trimming loses count.  The joint optimum keeps the whole group
    # (A0->t0, A1->t1) and the far singles; the group must read "allocated".
    return {
        "arms": [
            {"id": "A0", "x": 0, "y": 0, "max_extension": 10},
            {"id": "A1", "x": 100, "y": 0, "max_extension": 100},
            {"id": "A2", "x": 300, "y": 0, "max_extension": 10},
            {"id": "A3", "x": 1000, "y": 0, "max_extension": 500},
            {"id": "A4", "x": 2000, "y": 0, "max_extension": 500},
            {"id": "A5", "x": 3000, "y": 0, "max_extension": 500},
        ],
        "targets": [
            {"id": "t0", "x": 5, "y": 0, "priority": 9},
            {"id": "t1", "x": 95, "y": 0, "priority": 1},
            {"id": "s1", "x": 105, "y": 0, "priority": 8},
            {"id": "s2", "x": 305, "y": 0, "priority": 8},
            {"id": "s3", "x": 1005, "y": 0, "priority": 8},
            {"id": "s4", "x": 2005, "y": 0, "priority": 8},
        ],
        "clearance": 1,
        "minimum_allocations": 5,
        "fibre_groups": [{"id": "G1", "target_a": "t0", "target_b": "t1"}],
    }


def group_unallocatable_payload():
    # Both G1 members are only reachable by A0, so the pair cannot be
    # allocated: the group is wholly unallocated and atomicity reduces the
    # maximum from 5 (one member + four singles) to 4.
    return {
        "arms": [
            {"id": "A0", "x": 0, "y": 0, "max_extension": 100},
            {"id": "A1", "x": 300, "y": 0, "max_extension": 10},
            {"id": "A2", "x": 400, "y": 0, "max_extension": 10},
            {"id": "A3", "x": 1000, "y": 0, "max_extension": 500},
            {"id": "A4", "x": 2000, "y": 0, "max_extension": 500},
            {"id": "A5", "x": 3000, "y": 0, "max_extension": 500},
        ],
        "targets": [
            {"id": "t0", "x": 5, "y": 0, "priority": 9},
            {"id": "t1", "x": 10, "y": 0, "priority": 9},
            {"id": "s1", "x": 305, "y": 0, "priority": 1},
            {"id": "s2", "x": 405, "y": 0, "priority": 1},
            {"id": "s3", "x": 1005, "y": 0, "priority": 1},
            {"id": "s4", "x": 2005, "y": 0, "priority": 1},
        ],
        "clearance": 1,
        "minimum_allocations": 5,
        "fibre_groups": [{"id": "G1", "target_a": "t0", "target_b": "t1"}],
    }



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

    # 4a) dual-fibre group kept whole by the joint optimization
    _, body = request("POST", "/api/v1/adjudicate", group_kept_payload())
    assert body["status"] == "ok", body
    assert body["limits"]["fibre_groups_submitted"] == 1
    g = body["fibre_groups"][0]
    assert g["group_id"] == "G1" and g["status"] == "allocated", g
    by_target = {m["target_id"]: m for m in g["members"]}
    assert by_target["t0"]["arm_id"] == "A0"
    assert by_target["t1"]["arm_id"] == "A1"
    assert by_target["t0"]["arm_id"] != by_target["t1"]["arm_id"]
    pairs = {p["target_id"]: p["arm_id"] for p in body["assignments"]}
    assert pairs["t0"] == "A0" and pairs["t1"] == "A1"
    assert body["objective"]["num_allocations"] == 5
    for rec in body["clearance"]["pair_evidence"]:
        assert rec["satisfied"] is True
    print("group kept whole OK: G1 allocated with distinct arms")

    # 4b) impossible pair -> group wholly unallocated + atomicity shortfall
    _, body = request("POST", "/api/v1/adjudicate",
                      group_unallocatable_payload())
    assert body["status"] == "minimum_not_met", body
    assert body["objective"]["num_allocations"] == 4, body["objective"]
    g = body["fibre_groups"][0]
    assert g["status"] == "unallocated", g
    assert all(not m["allocated"] and m["arm_id"] is None
               for m in g["members"])
    df = body["witness"]["dual_fibre"]
    assert df["constraint"] == "dual_fibre_atomic"
    assert df["maximum_without_atomicity"] == 5
    assert df["maximum_with_atomicity"] == 4
    assert df["allocations_blocked_by_atomicity"] == 1
    assert df["would_be_split_groups"][0]["group_id"] == "G1"
    assert "atomic" in body["reason"]
    for rec in body["clearance"]["pair_evidence"]:
        assert rec["satisfied"] is True
    print("group dropped whole OK: max=4 with dual-fibre shortfall evidence")

    # 4c) malformed groups rejected with 422
    bad = group_kept_payload()
    bad["fibre_groups"] = [
        {"id": "G1", "target_a": "t0", "target_b": "t0"}]  # same target
    request("POST", "/api/v1/adjudicate", bad, expect=(422,))
    bad = group_kept_payload()
    bad["fibre_groups"] = [
        {"id": "G1", "target_a": "t0", "target_b": "missing"}]
    request("POST", "/api/v1/adjudicate", bad, expect=(422,))
    bad = group_kept_payload()
    bad["fibre_groups"] = [
        {"id": "G1", "target_a": "t0", "target_b": "t1"},
        {"id": "G2", "target_a": "t0", "target_b": "t2"}]  # t0 reused
    request("POST", "/api/v1/adjudicate", bad, expect=(422,))
    print("malformed fibre groups rejected with 422")

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
