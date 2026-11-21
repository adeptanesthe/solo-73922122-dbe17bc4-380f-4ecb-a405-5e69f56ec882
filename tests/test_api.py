"""API 层测试：正常流程、页面服务与整批拒绝行为。"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "measured_times": [0, 1, 2, 3, 5, 8, 13],
    "measured_values": [25.3, 196.1, 330.8, 433.5, 578.2, 695.1, 769.9],
    "candidate_times": [0.5, 1.5, 2.5, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,
                        8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 14, 14.5, 15, 15.5],
    "length_scale": 2.0,
    "noise_std": 5.0,
    "signal_variance": 100000.0,
    "min_interval": 1.0,
}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "高温炉温度采样选点" in r.text


def test_select_ok():
    r = client.post("/api/select", json=VALID_PAYLOAD)
    assert r.status_code == 200, r.text
    data = r.json()
    # 选中两个不同的候选时刻
    assert len(data["selected_times"]) == 2
    assert data["selected_times"][0] != data["selected_times"][1]
    assert len(data["selected_indices"]) == 2
    # 总量自洽
    assert data["total_variance_before"] > data["total_variance_after"]
    assert abs(data["total_variance_before"] - data["total_variance_after"]
               - data["total_reduction"]) < 1e-6 * data["total_variance_before"]
    # 候选明细完整且选后方差不超过选前
    cands = data["candidates"]
    assert len(cands) == len(VALID_PAYLOAD["candidate_times"])
    for c in cands:
        assert c["variance_after"] <= c["variance_before"] + 1e-6
    # 已测时刻 5.0 对应的候选点被排除
    c5 = next(c for c in cands if c["time"] == 5.0)
    assert not c5["eligible"] and c5["exclusion_reason"]
    # 选中的点必须合法
    for idx in data["selected_indices"]:
        assert cands[idx]["eligible"] and cands[idx]["selected"]
    assert 0 < data["n_pairs_feasible"] <= data["n_pairs_total"]


def test_nan_in_candidates_rejected():
    payload = dict(VALID_PAYLOAD)
    raw = json.dumps(payload).replace("0.5", "NaN", 1)  # 构造含 NaN 的非法 JSON 字面量
    r = client.post("/api/select", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] in ("INVALID_INPUT", "NON_FINITE_INPUT")


def test_non_psd_kernel_rejected():
    payload = dict(VALID_PAYLOAD, measured_times=[1.0, 1.0],
                   measured_values=[100.0, 100.0], noise_std=0.0)
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "KERNEL_NOT_PSD"


def test_no_feasible_combination_rejected():
    payload = dict(VALID_PAYLOAD, min_interval=1e6)
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "NO_FEASIBLE_COMBINATION"


def test_mismatched_measured_lengths_rejected():
    payload = dict(VALID_PAYLOAD, measured_values=[1.0, 2.0])
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_invalid_parameters_rejected():
    for field, bad in [("length_scale", 0.0), ("noise_std", -1.0),
                       ("signal_variance", -2.0), ("min_interval", -0.5)]:
        r = client.post("/api/select", json=dict(VALID_PAYLOAD, **{field: bad}))
        assert r.status_code == 422, field
        assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_error_response_has_no_partial_result():
    payload = dict(VALID_PAYLOAD, min_interval=1e6)
    r = client.post("/api/select", json=payload)
    body = r.json()
    assert "candidates" not in body and "selected_times" not in body
