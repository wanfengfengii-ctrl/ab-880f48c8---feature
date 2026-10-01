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


# ----------------------------------------------------------------------
# Dual-fibre groups
# ----------------------------------------------------------------------
GROUPABLE_PAYLOAD = {
    "arms": [
        {"id": f"A{i}", "x": i * 1000, "y": 0, "max_extension": 500}
        for i in range(6)
    ],
    "targets": [
        {"id": f"T{i}", "x": i * 1000 + 10, "y": 0, "priority": i + 1}
        for i in range(6)
    ],
    "clearance": 5,
    "minimum_allocations": 6,
}


def test_ungrouped_response_has_no_group_keys(client):
    r = client.post("/api/v1/adjudicate", json=GROUPABLE_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "dual_fibre_groups" not in body
    assert "dual_fibre_groups_submitted" not in body["limits"]
    # feasible responses carry no witness at all; shortfall witnesses in
    # ungrouped requests must not carry group-only keys either
    shortfall = client.post(
        "/api/v1/adjudicate",
        json={**GROUPABLE_PAYLOAD, "minimum_allocations": 7}).json()
    assert shortfall["status"] == "minimum_not_met"
    assert "ungrouped_max_allocations" not in shortfall["witness"]
    assert "dual_fibre_shortfall" not in shortfall["witness"]


def test_group_allocated_as_a_pair_with_two_arms(client):
    payload = dict(GROUPABLE_PAYLOAD)
    payload["dual_fibre_groups"] = [
        {"target_a": "T0", "target_b": "T1"},
        {"target_a": "T2", "target_b": "T3"},
    ]
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    groups = body["dual_fibre_groups"]
    assert len(groups) == 2
    for g in groups:
        assert g["allocated"] is True
        assert len(g["allocations"]) == 2
        assert len({a["arm_id"] for a in g["allocations"]}) == 2
        assert {a["target_id"] for a in g["allocations"]} == set(g["targets"])
    assert body["limits"]["dual_fibre_groups_submitted"] == 2
    # no half-groups anywhere in the flat assignment view
    assigned_targets = {p["target_id"] for p in body["assignments"]}
    for g in groups:
        assert set(g["targets"]) <= assigned_targets


def test_group_validation_rejects_bad_groups(client):
    def post(groups, base=None):
        payload = dict(base or GROUPABLE_PAYLOAD)
        payload["dual_fibre_groups"] = groups
        return client.post("/api/v1/adjudicate", json=payload)

    # same target twice in one group
    assert post([{"target_a": "T0", "target_b": "T0"}]).status_code == 422
    # unknown target reference
    assert post([{"target_a": "T0", "target_b": "Nope"}]).status_code == 422
    # target reused across two groups
    assert post([
        {"target_a": "T0", "target_b": "T1"},
        {"target_a": "T1", "target_b": "T2"},
    ]).status_code == 422
    # seven groups exceed the 6-group cap (needs 14 targets; use big payload)
    big = {
        "arms": [{"id": f"A{i}", "x": i * 30, "y": 1000, "max_extension": 20}
                 for i in range(12)],
        "targets": [{"id": f"G{j}", "x": j * 20, "y": 1000, "priority": 1}
                    for j in range(16)],
        "clearance": 1,
        "minimum_allocations": 1,
        "dual_fibre_groups": [
            {"target_a": f"G{2*g}", "target_b": f"G{2*g+1}"} for g in range(7)
        ],
    }
    assert client.post("/api/v1/adjudicate", json=big).status_code == 422
    # empty group list rejected (must be omitted or 1..6 groups)
    assert post([]).status_code == 422
    # extra field inside a group rejected under strict model
    assert post([{"target_a": "T0", "target_b": "T1", "x": 1}]).status_code == 422


def test_group_collision_shortfall_reports_atomicity(client):
    # Two arms share a base (always clash); grouping their two targets makes
    # the group uncompletable, and neither target may be taken alone.
    arms = [{"id": "A0", "x": 0, "y": 0, "max_extension": 200},
            {"id": "A1", "x": 0, "y": 0, "max_extension": 200}] + \
           [{"id": f"A{i}", "x": i * 1000, "y": 0, "max_extension": 200}
            for i in range(2, 6)]
    targets = [{"id": "T0", "x": 10, "y": 0, "priority": 5},
               {"id": "T1", "x": 12, "y": 0, "priority": 9}] + \
              [{"id": f"F{j}", "x": j * 1000 + 10, "y": 0, "priority": 5}
               for j in range(2, 6)]
    payload = {"arms": arms, "targets": targets, "clearance": 1,
               "minimum_allocations": 6,
               "dual_fibre_groups": [{"target_a": "T0", "target_b": "T1"}]}
    r = client.post("/api/v1/adjudicate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "minimum_not_met"
    assert body["objective"]["num_allocations"] == 4
    g = body["dual_fibre_groups"][0]
    assert g["allocated"] is False and g["allocations"] == []
    d = body["witness"]["dual_fibre_shortfall"]
    assert d["groups_submitted"] == 1 and d["groups_allocated"] == 0
    assert d["grouped_max_allocations"] == 4
    assert d["ungrouped_max_allocations"] == 5
    assert d["allocations_lost_to_atomicity"] == 1
    # same instance without groups: joint optimum is 5 and is feasible? No -
    # minimum 6 is still unmet, but the reported maximum must be 5.
    plain = dict(payload)
    plain.pop("dual_fibre_groups")
    body2 = client.post("/api/v1/adjudicate", json=plain).json()
    assert body2["objective"]["num_allocations"] == 5
    assert "all-or-nothing group constraint" in body["reason"]
    # returned partial allocation remains collision free
    for p in body["clearance"]["pair_evidence"]:
        assert p["satisfied"] is True


def test_group_completion_changes_joint_choice(client):
    # Local arms A0/A1 and targets a (A0-only), b (A1-only), c (both).
    # Ungrouped optimum takes a+c; group (a,b) forces completion a+b instead
    # of deleting the residual, keeping the full count.
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
    payload = {"arms": arms, "targets": targets, "clearance": 1,
               "minimum_allocations": 6,
               "dual_fibre_groups": [{"target_a": "a", "target_b": "b"}]}
    body = client.post("/api/v1/adjudicate", json=payload).json()
    assert body["status"] == "ok"
    by_target = {p["target_id"]: p["arm_id"] for p in body["assignments"]}
    assert by_target["a"] == "A0" and by_target["b"] == "A1"
    assert "c" not in by_target
    g = body["dual_fibre_groups"][0]
    assert g["allocated"] is True
