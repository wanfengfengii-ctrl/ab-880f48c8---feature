"""End-to-end API tests: validation gating, adjudication and evidence."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def far_apart_payload(**overrides):
    payload = {
        "arms": [
            {"id": f"A{i}", "x": i * 100, "y": 0, "max_extension": 500}
            for i in range(6)
        ],
        "targets": [
            {"id": f"T{i}", "x": i * 100 + 10, "y": 10, "priority": i + 1}
            for i in range(6)
        ],
        "clearance": 5,
        "minimum_allocations": 6,
    }
    payload.update(overrides)
    return payload


def test_readiness_passed_startup_self_check(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert client.get("/health/live").status_code == 200


def test_valid_request_returns_optimal_assignment_with_evidence(client):
    r = client.post("/api/v1/adjudicate", json=far_apart_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["feasible"] is True
    assert body["objective"]["num_allocations"] == 6
    assert body["objective"]["priority_sum"] == 21
    assert len(body["assignments"]) == 6
    assert body["unassigned_arms"] == []
    assert body["unassigned_targets"] == []
    # every submitted arm has a reported length
    assert set(body["arm_lengths"]) == {f"A{i}" for i in range(6)}
    # pairwise clearance evidence: 15 pairs, all satisfied
    evidence = body["clearance"]["pair_evidence"]
    assert len(evidence) == 15
    for p in evidence:
        assert p["satisfied"] is True
        assert p["distance"] + 1e-9 >= p["required_clearance"]
        assert p["witness"]["point_on_first"]
    assert body["clearance"]["all_pairs_satisfied"] is True


def test_minimum_not_met_reports_maximum_and_reason(client):
    # Three groups; within each group both pairings of two arms/targets
    # violate clearance 3 (converging gap 2 / crossing), so at most one
    # allocation per group -> maximum 3 although 6 are requested.
    arms, targets = [], []
    for g in range(3):
        gx = 200 * g
        arms += [{"id": f"L{g}", "x": gx, "y": 0, "max_extension": 100},
                 {"id": f"R{g}", "x": gx, "y": 10, "max_extension": 100}]
        targets += [{"id": f"lo{g}", "x": gx + 10, "y": 4, "priority": 5 + g},
                    {"id": f"hi{g}", "x": gx + 10, "y": 6, "priority": 1 + g}]
    payload = {"arms": arms, "targets": targets,
               "clearance": 3, "minimum_allocations": 6}
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "minimum_not_met"
    assert body["feasible"] is False
    assert body["objective"]["num_allocations"] == 3
    assert "minimum_allocations=6" in body["reason"]
    assert "3" in body["reason"]
    assert body["witness"]["blocking_pair"] is not None
    bp = body["witness"]["blocking_pair"]
    assert bp["distance"] < bp["required_clearance"]
    # the returned partial assignment is itself collision free
    for p in body["clearance"]["pair_evidence"]:
        assert p["satisfied"] is True


def test_invalid_inputs_are_rejected_before_solving(client):
    bad_cases = [
        {"arms": far_apart_payload()["arms"][:5]},  # too few arms
        {"clearance": 0},
        {"clearance": -2},
        {"minimum_allocations": 0},
    ]
    for patch in bad_cases:
        payload = far_apart_payload()
        payload.update(patch)
        r = client.post("/api/v1/adjudicate", json=payload)
        assert r.status_code == 422, (patch, r.status_code)

    # non-positive priority embedded in a target
    payload = far_apart_payload()
    payload["targets"][0]["priority"] = 0
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422

    # duplicate target ids
    payload = far_apart_payload()
    payload["targets"][1]["id"] = payload["targets"][0]["id"]
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422

    # float where integer base coordinate is required
    payload = far_apart_payload()
    payload["arms"][0]["x"] = 1.5
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422

    # booleans must not sneak in as integers
    payload = far_apart_payload()
    payload["clearance"] = True
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422
    payload = far_apart_payload()
    payload["targets"][0]["priority"] = False
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422

    # unknown field
    payload = far_apart_payload()
    payload["bogus"] = 1
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422


def test_too_many_targets_rejected(client):
    payload = far_apart_payload()
    payload["targets"] = [
        {"id": f"T{i}", "x": i, "y": 0, "priority": 1} for i in range(17)
    ]
    assert client.post("/api/v1/adjudicate", json=payload).status_code == 422


def test_maximum_size_instance_accepted(client):
    payload = {
        "arms": [{"id": f"A{i}", "x": i * 40, "y": 0, "max_extension": 45}
                 for i in range(12)],
        "targets": [{"id": f"T{j}", "x": j * 30, "y": 6,
                     "priority": (j % 9) + 1} for j in range(16)],
        "clearance": 2,
        "minimum_allocations": 1,
    }
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["objective"]["num_allocations"] >= 1
    for p in body["clearance"]["pair_evidence"]:
        assert p["distance"] + 1e-9 >= p["required_clearance"]


# ---------------------------------------------------------------------------
# Dual-fibre observation groups
# ---------------------------------------------------------------------------
def group_payload(**overrides):
    # A0/A1 cover the grouped pair T0/T1; A2..A5 cover singles T2..T5.
    return far_apart_payload(
        fibre_groups=[{"id": "G0", "target_a": "T0", "target_b": "T1"}])


def test_request_without_groups_is_unchanged(client):
    body = far_apart_payload()
    assert "fibre_groups" not in body
    r = client.post("/api/v1/adjudicate", json=body)
    assert r.status_code == 200
    assert r.json()["fibre_groups"] is None
    assert r.json()["limits"]["fibre_groups_submitted"] == 0


def test_group_allocated_with_arms_per_target(client):
    r = client.post("/api/v1/adjudicate", json=group_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["limits"]["fibre_groups_submitted"] == 1
    groups = body["fibre_groups"]
    assert len(groups) == 1
    g = groups[0]
    assert g["group_id"] == "G0" and g["status"] == "allocated"
    by_target = {m["target_id"]: m for m in g["members"]}
    assert set(by_target) == {"T0", "T1"}
    assert by_target["T0"]["allocated"] and by_target["T1"]["allocated"]
    assert by_target["T0"]["arm_id"] == "A0"
    assert by_target["T1"]["arm_id"] == "A1"
    assert by_target["T0"]["arm_id"] != by_target["T1"]["arm_id"]
    for m in g["members"]:
        assert m["extension"] > 0 and m["extension_sq"] > 0


def test_group_jointly_kept_not_trimmed_from_old_optimum(client):
    # Ungrouped optimum takes T0 (A0), s1 (A1), s2 (A2): group G1=(t0,t1)
    # would be split, and trimming leaves 2.  The joint optimum keeps the
    # whole group and still allocates 3.
    payload = {
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
        "minimum_allocations": 3,
        "fibre_groups": [{"id": "G1", "target_a": "t0", "target_b": "t1"}],
    }
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok", body["reason"]
    pairs = {p["target_id"]: p["arm_id"] for p in body["assignments"]}
    assert pairs["t0"] == "A0" and pairs["t1"] == "A1" and pairs["s2"] == "A2"
    # the whole group plus the three far singles
    assert body["objective"]["num_allocations"] == 5
    g = body["fibre_groups"][0]
    assert g["status"] == "allocated"


def test_group_reported_whole_unallocated_with_atomicity_shortfall(client):
    # G1 members are both only reachable by A0: the pair is impossible, so
    # the group is wholly unallocated and the atomic shortfall is explained.
    payload = {
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
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "minimum_not_met"
    # four unrelated singles remain achievable; the global maximum is 4
    assert body["objective"]["num_allocations"] == 4
    g = body["fibre_groups"][0]
    assert g["status"] == "unallocated"
    assert all(not m["allocated"] and m["arm_id"] is None
               for m in g["members"])
    df = body["witness"]["dual_fibre"]
    assert df["constraint"] == "dual_fibre_atomic"
    assert df["maximum_without_atomicity"] == 5
    assert df["maximum_with_atomicity"] == 4
    assert df["allocations_blocked_by_atomicity"] == 1
    assert df["would_be_split_groups"][0]["group_id"] == "G1"
    assert "atomic" in body["reason"]


def test_invalid_group_payloads_rejected(client):
    def post(patch):
        payload = group_payload()
        payload.update(patch)
        return client.post("/api/v1/adjudicate", json=payload)

    # two distinct targets required
    g = [{"id": "G0", "target_a": "T0", "target_b": "T0"}]
    assert post({"fibre_groups": g}).status_code == 422
    # unknown target reference
    g = [{"id": "G0", "target_a": "T0", "target_b": "NOPE"}]
    assert post({"fibre_groups": g}).status_code == 422
    # target reused across groups
    g = [
        {"id": "G0", "target_a": "T0", "target_b": "T1"},
        {"id": "G1", "target_a": "T0", "target_b": "T2"},
    ]
    assert post({"fibre_groups": g}).status_code == 422
    # duplicate group ids
    g = [
        {"id": "G0", "target_a": "T0", "target_b": "T1"},
        {"id": "G0", "target_a": "T2", "target_b": "T3"},
    ]
    assert post({"fibre_groups": g}).status_code == 422
    # more than six groups
    g = [{"id": f"G{i}", "target_a": f"T{2*i}", "target_b": f"T{2*i+1}"}
         for i in range(7)]
    assert post({"fibre_groups": g}).status_code == 422
    # empty group list is rejected (field is optional, not a list of zero)
    assert post({"fibre_groups": []}).status_code == 422
    # unknown field inside a group
    g = [{"id": "G0", "target_a": "T0", "target_b": "T1", "x": 1}]
    assert post({"fibre_groups": g}).status_code == 422

